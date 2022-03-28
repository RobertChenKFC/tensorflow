/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/compiler/mlir/lite/tf_tfl_passes.h"

#include <string>

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/transforms/passes.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_config.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_passes.h"
#include "tensorflow/compiler/mlir/lite/quantization/tensorflow/passes.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/lite/utils/fake_quant_utils.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops_a_m.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_saved_model_passes.h"
#include "tensorflow/compiler/mlir/utils/name_utils.h"

namespace MyPass {
CalibrationData::CalibrationData(const std::string &path,
                                 const std::string &filename) : path_(path) {
  std::fstream file(path + "/" + filename);
  std::string opName;
  while (file >> opName) {
    float intercept, alphaHat;
    file >> intercept >> alphaHat;
    std::vector<float> betaHats, breakpoints;
    int n;
    file >> n;
    while (n--) {
      float betaHat;
      file >> betaHat;
      betaHats.push_back(betaHat);
    }
    file >> n;
    while (n--) {
      float breakpoint;
      file >> breakpoint;
      breakpoints.push_back(breakpoint);
    }
    data_[opName] = std::make_tuple(intercept, alphaHat, betaHats, breakpoints);
  }
}

CalibrationData::Coeffs CalibrationData::getCoeffs(mlir::Location loc) const {
  auto FusedLocToName = [](mlir::FusedLoc fusedLoc) {
    auto nameLoc = fusedLoc.getLocations()[1].dyn_cast<mlir::NameLoc>();
    auto name = nameLoc.getName().str();
    auto pos = name.find_first_of('@');
    if (pos != std::string::npos)
      name = name.substr(0, pos);
    return name;
  };

  mlir::FusedLoc fusedLoc;
  std::string opName;
  auto i = 0;
  while (!(fusedLoc = loc.dyn_cast<mlir::FusedLoc>())) {
    mlir::CallSiteLoc callSiteLoc;
    if (callSiteLoc = loc.dyn_cast<mlir::CallSiteLoc>()) {
      loc = callSiteLoc.getCallee();
    } else {
      llvm::errs() << "ERROR: location name not found for " << loc << "\n";
      throw std::runtime_error("Location name not found");
    }
    auto fusedLoc = callSiteLoc.getCaller().dyn_cast<mlir::FusedLoc>();
    auto name = FusedLocToName(fusedLoc);
    if (i > 0) {
      if (i == 1)
        opName = name;
      else
        opName += "/" + name;
    }
    ++i;
  }
  auto name = FusedLocToName(fusedLoc);
  if (i == 1)
    opName = name;
  else
    opName += "/" + name;

  auto it = data_.find(opName);
  if (it == data_.end()) {
    llvm::errs() << "ERROR: approximation data not found for "
                 << opName << "\n";
    throw std::runtime_error("Approximation data not found");
  }
  return it->second;
}

ReplaceAbsOp::ReplaceAbsOp(mlir::MLIRContext *ctx)
    : mlir::OpRewritePattern<mlir::TFL::AbsOp>(ctx, /*benefit=*/1) {}

mlir::LogicalResult ReplaceAbsOp::matchAndRewrite(
    mlir::TFL::AbsOp op, mlir::PatternRewriter &rewriter) const {
  llvm::dbgs() << "INFO: AbsOp is called!\n";

  auto x = op.getOperand();
  auto xType = x.getType().dyn_cast<mlir::ShapedType>();
  auto zeroType = mlir::RankedTensorType::get(xType.getShape(),
                                              xType.getElementType());
  auto zeroAttr = mlir::SplatElementsAttr::get(zeroType, 0);
  auto zero = rewriter.create<mlir::TFL::ConstOp>(op.getLoc(), zeroType,
                                                  zeroAttr);
  auto negX = rewriter.create<mlir::TFL::SubOp>(
      op.getLoc(), zero, x, rewriter.getStringAttr("NONE"));
  auto max = rewriter.replaceOpWithNewOp<mlir::TFL::MaximumOp>(op, x, negX);

  return mlir::success();
}

ReplaceLeakyReluOp::ReplaceLeakyReluOp(mlir::MLIRContext *ctx)
    : mlir::OpRewritePattern<mlir::TFL::LeakyReluOp>(ctx, /*benefit=*/1) {}

mlir::LogicalResult ReplaceLeakyReluOp::matchAndRewrite(
    mlir::TFL::LeakyReluOp op,
    mlir::PatternRewriter &rewriter) const {
  llvm::dbgs() << "INFO: LeakyReluOp is called!\n";

  auto x = op.getOperand();
  auto resultType = op.getResult().getType();

#ifdef LEAKY_RELU_RELU
  rewriter.replaceOpWithNewOp<mlir::TFL::ReluOp>(op, x);
#endif

#if defined(LEAKY_RELU_PRELU) || defined(LEAKY_RELU_X_ALPHAX) || \
    defined(LEAKY_RELU_POS_NEG)
  auto alphaOldAttr = op->getAttr("alpha");
  auto alpha = alphaOldAttr.dyn_cast<mlir::FloatAttr>().getValue()
      .convertToFloat();
  auto xType = x.getType().dyn_cast<mlir::ShapedType>();
  auto alphaShape = xType.getShape().vec();
  alphaShape.erase(alphaShape.begin());
  for (auto &dim : alphaShape)
    dim = 1;
  auto alphaType = xType.clone(alphaShape);
  auto alphaAttr = mlir::SplatElementsAttr::get(alphaType, alpha);
  auto alphaConst = rewriter.create<mlir::TFL::ConstOp>(
      op.getLoc(), alphaAttr.getType(), alphaAttr);
#endif

#ifdef LEAKY_RELU_PRELU
  // Replace leaky-relu with PRelu
  rewriter.replaceOpWithNewOp<mlir::TFL::PReluOp>(op, resultType, x, alphaConst);
#endif

#ifdef LEAKY_RELU_X_ALPHAX
  // Replace leaky-relu with max(x, alpha * x); this is correct only if
  // 0 < alpha < 1
  auto noActivationAttr = rewriter.getStringAttr("NONE");
  auto alphaX = rewriter.create<mlir::TFL::MulOp>(
      op.getLoc(), alphaConst, x, noActivationAttr);
  rewriter.replaceOpWithNewOp<mlir::TFL::MaximumOp>(op, x, alphaX);
#endif

#ifdef LEAKY_RELU_POS_NEG
  // Replace leaky-relu with relu(x) - alpha * relu(-x)
  auto pos = rewriter.create<mlir::TFL::ReluOp>(op.getLoc(), x);
  auto zeroAttr = mlir::SplatElementsAttr::get(alphaType, 0);
  auto zero = rewriter.create<mlir::TFL::ConstOp>(
      op.getLoc(), zeroAttr.getType(), zeroAttr);
  auto noActivationAttr = rewriter.getStringAttr("NONE");
  auto negX = rewriter.create<mlir::TFL::SubOp>(
      op.getLoc(), zero, x, noActivationAttr);
  auto reluNegX = rewriter.create<mlir::TFL::ReluOp>(op.getLoc(), negX);
  auto neg = rewriter.create<mlir::TFL::MulOp>(
      op.getLoc(), alphaConst, reluNegX, noActivationAttr);
  auto newOp = rewriter.replaceOpWithNewOp<mlir::TFL::SubOp>(
      op, pos, neg, noActivationAttr);
#endif

  return mlir::success();
}

ReplaceTileOp::ReplaceTileOp(mlir::MLIRContext *ctx)
    : mlir::OpRewritePattern<mlir::TFL::TileOp>(ctx, /*benefit=*/1) {}

mlir::LogicalResult ReplaceTileOp::matchAndRewrite(
    mlir::TFL::TileOp op, mlir::PatternRewriter &rewriter) const {
  llvm::dbgs() << "INFO: TileOp is called!\n";

  auto x = op.getOperand(0);
  auto xType = x.getType().dyn_cast<mlir::ShapedType>();
  auto xShape = xType.getShape().vec();

  auto multiplesOp = mlir::dyn_cast<mlir::ConstantOp>(
      op.getOperand(1).getDefiningOp());
  auto multiplesAttr = multiplesOp->getAttr("value").dyn_cast<
      mlir::DenseElementsAttr>();

  int axis = 0;
  auto result = x;
  llvm::SmallVector<mlir::Value, 5> xs;
  for (auto multiple : multiplesAttr.getValues<int>()) {
    if (multiple > 1) {
      xs.clear();
      for (uint64_t i = 0; i < multiple; ++i)
        xs.push_back(result);
      xShape[axis] *= multiple;
      auto concatType = mlir::RankedTensorType::get(
          xShape, xType.getElementType());
      result = rewriter.create<mlir::TFL::ConcatenationOp>(
          op.getLoc(), concatType, xs, axis, "NONE");
    }
    ++axis;
  }
  rewriter.replaceOp(op, {result});

  return mlir::success();
}

ReplaceTFLUnpackOp::ReplaceTFLUnpackOp(mlir::MLIRContext *ctx)
    : mlir::OpRewritePattern<mlir::TFL::UnpackOp>(ctx, /*benefit=*/1) {}

mlir::LogicalResult ReplaceTFLUnpackOp::matchAndRewrite(
    mlir::TFL::UnpackOp op, mlir::PatternRewriter &rewriter) const {
  llvm::dbgs() << "INFO: UnpackOp is called!\n";

  auto x = op.getOperand();
  auto xType = x.getType().dyn_cast<mlir::ShapedType>();

  auto numAttr = op->getAttr("num").dyn_cast<mlir::IntegerAttr>();
  auto num = numAttr.getInt();
  auto axisAttr = op->getAttr("axis").dyn_cast<mlir::IntegerAttr>();
  auto axis = axisAttr.getInt();

  auto size = static_cast<int64_t>(xType.getShape().size());
  auto stridedSliceInputShape = llvm::ArrayRef<int64_t>(size);
  auto stridedSliceInputType = mlir::RankedTensorType::get(
      stridedSliceInputShape, rewriter.getI32Type());
  auto strideAttr = mlir::SplatElementsAttr::get(stridedSliceInputType, 1);
  auto strideConst = rewriter.create<mlir::TFL::ConstOp>(
      op.getLoc(), strideAttr.getType(), strideAttr);

  auto slicedXShape = xType.getShape().vec();
  slicedXShape.erase(slicedXShape.begin() + axis);
  auto slicedXType =
      mlir::RankedTensorType::get(slicedXShape, xType.getElementType());

  auto zerosMask = rewriter.getI32IntegerAttr(0);
  auto shrinkAxisMask = rewriter.getI32IntegerAttr(1 << axis);

  std::vector<mlir::Value> slicedXs;
  slicedXs.reserve(num);
  for (long i = 0; i < num; ++i) {
    auto beginVals = std::vector<int32_t>(xType.getShape().size(), 0);
    beginVals[axis] = i;
    auto beginAttr = rewriter.getI32TensorAttr(beginVals);
    auto beginConst = rewriter.create<mlir::TFL::ConstOp>(
        op.getLoc(), beginAttr.getType(), beginAttr);

    // TODO: check if the vector can be reused
    auto endVals = std::vector<int32_t>();
    endVals.reserve(xType.getShape().size());
    for (auto dim : xType.getShape())
      endVals.push_back(static_cast<int32_t>(dim));
    endVals[axis] = i + 1;
    auto endAttr = rewriter.getI32TensorAttr(endVals);
    auto endConst = rewriter.create<mlir::TFL::ConstOp>(
        op.getLoc(), endAttr.getType(), endAttr);

    auto slicedX = rewriter.create<mlir::TFL::StridedSliceOp>(
        op.getLoc(), slicedXType, x, beginConst, endConst, strideConst,
        zerosMask, zerosMask, zerosMask, zerosMask, shrinkAxisMask);
    slicedXs.push_back(slicedX);
  }

  rewriter.replaceOp(op, slicedXs);

  return mlir::success();
}

#ifndef EXP_DISABLE
ReplaceExpOp::ReplaceExpOp(mlir::MLIRContext *ctx)
    : mlir::OpRewritePattern<mlir::TFL::ExpOp>(ctx, /*benefit=*/1) {}

mlir::LogicalResult ReplaceExpOp::matchAndRewrite(
    mlir::TFL::ExpOp op, mlir::PatternRewriter &rewriter) const {
  llvm::dbgs() << "INFO: ExpOp is called!\n";

  auto x = op.getOperand();

#ifdef EXP_LEAST_SQ
  // least square in range [-3, 3]
  float coeffs[] = { 0.59803783, 0.7892836, 0.91375127, 0.26916787 };
#endif
#ifdef EXP_REL_LEAST_SQ
  // least square relative in range [-3, 3]
  float coeffs[] = { 0.90509352, 1.31294584, 0.78952683, 0.15414826 };
#endif
#ifdef EXP_MINIMAX
  // minimax approximation in range [-3, 3]
  float coeffs[] = { 0.3986013, 0.52297465, 0.99602537, 0.31292411 };
#endif
#ifdef EXP_TAYLOR
  // taylor polynomial centered around 0
  float coeffs[] = { 1.0, 1.0, 0.5, 0.16666666666666666 };
#endif
  auto sum = PolynomialValueTFL(rewriter, op.getLoc(), x, coeffs, 3);
  rewriter.replaceOp(op, {sum});

  return mlir::success();
}
#endif

#ifndef LOG_DISABLE
ReplaceLogOp::ReplaceLogOp(mlir::MLIRContext *ctx)
    : mlir::OpRewritePattern<mlir::TFL::LogOp>(ctx, /*benefit=*/1) {}

mlir::LogicalResult ReplaceLogOp::matchAndRewrite(
    mlir::TFL::LogOp op, mlir::PatternRewriter &rewriter) const {
  llvm::dbgs() << "INFO: LogOp is called!\n";

  auto x = op.getOperand();

#ifdef LOG_LEAST_SQ
  // least square in range [exp(-3)+1, exp(3)+1]
  float coeffs[] = { -0.10349343, 0.44997741, -0.02630398, 0.00058021 };
#endif
#ifdef LOG_REL_LEAST_SQ
  // least square relative in range [exp(-3)+1, exp(3)+1]
  float coeffs[] = { -0.3548069 , 0.54611009, -0.03581849, 0.00085092 };
#endif
#ifdef LOG_MINIMAX
  // minimax approximation in range [exp(-3)+1, exp(3)+1]
  float coeffs[] = { -0.39904023, 0.57170487, -0.03831855, 0.00091105 };
#endif
#ifdef LOG_TAYLOR
  // taylor polynomial centered around ((exp(-3) + 1) + exp(3) + 1) / 2
  float coeffs[] = { 0.5706941892542055, 0.2710599583854728,
                     -0.012245583506655708, 0.0002458731374607354 };
#endif
  auto sum = PolynomialValueTFL(rewriter, op.getLoc(), x, coeffs, 3);
  rewriter.replaceOp(op, {sum});

  return mlir::success();
}
#endif

ReplaceFullyConnectedOp::ReplaceFullyConnectedOp(mlir::MLIRContext *ctx)
    : mlir::OpRewritePattern<mlir::TFL::FullyConnectedOp>(ctx, /*benefit=*/1) {}

mlir::LogicalResult ReplaceFullyConnectedOp::matchAndRewrite(
    mlir::TFL::FullyConnectedOp op,
    mlir::PatternRewriter &rewriter) const {
  llvm::dbgs() << "INFO: FullyConnectedOp is called!\n";

  auto input = op.getOperand(0);
  auto inputType = input.getType().dyn_cast<mlir::ShapedType>();
  auto inputShape = inputType.getShape();

#ifdef FULLY_CONNECTED_NO_KEEP_DIMS
  if (op->getAttrOfType<mlir::BoolAttr>("keep_num_dims").getValue()) {
    rewriter.replaceOpWithNewOp<mlir::TFL::FullyConnectedOp>(
        op, op.getResult(0).getType(), op.getOperand(0),
        op.getOperand(1), op.getOperand(2),
        op->getAttrOfType<mlir::StringAttr>("fused_activation_function"),
        op->getAttrOfType<mlir::StringAttr>("weights_format"),
        rewriter.getBoolAttr(false));

    llvm::dbgs() << "Keep num dims was originally true.\n";

    return mlir::success();
  }
#endif

  if (inputShape.size() == 3 && inputShape.front() > 1) {
    auto batchSize = inputShape.front();
    auto splitDimNum = static_cast<int32_t>(0);
    auto splitDimArr = llvm::ArrayRef<int32_t>(splitDimNum);
    auto splitDimAttr = rewriter.getI32TensorAttr(splitDimArr);
    auto splitDim = rewriter.create<mlir::TFL::ConstOp>(
        op.getLoc(), splitDimAttr.getType(), splitDimAttr);

    auto splittedShapeVec = std::vector<int64_t>(inputShape.vec());
    splittedShapeVec.front() = 1;
    auto splittedShape = llvm::ArrayRef<int64_t>(splittedShapeVec);
    auto splittedType = mlir::RankedTensorType::get(
        splittedShape, inputType.getElementType());
    auto splittedTypesVec = std::vector<mlir::Type>(batchSize, splittedType);
    auto splittedTypes = llvm::ArrayRef<mlir::Type>(splittedTypesVec);
    auto splittedInputs = rewriter.create<mlir::TFL::SplitOp>(
        op.getLoc(), splittedTypes, splitDim, input, batchSize);

    auto outputType = op.getResult(0).getType().dyn_cast<mlir::ShapedType>();
    auto outputShape = outputType.getShape();
    auto resultShapeVec = std::vector<int64_t>(outputShape.vec());
    resultShapeVec.front() = 1;
    auto resultShape = llvm::ArrayRef<int64_t>(resultShapeVec);
    auto resultType = mlir::RankedTensorType::get(
        resultShape, outputType.getElementType());

    auto resultsVec = std::vector<mlir::Value>();
    resultsVec.reserve(batchSize);
    for (auto splittedInput : splittedInputs.getResults()) {
      auto result = rewriter.create<mlir::TFL::FullyConnectedOp>(
          op.getLoc(), resultType, splittedInput,
          op.getOperand(1), op.getOperand(2),
          op->getAttrOfType<mlir::StringAttr>("fused_activation_function"),
          op->getAttrOfType<mlir::StringAttr>("weights_format"),
          op->getAttrOfType<mlir::BoolAttr>("keep_num_dims"),
          op->getAttrOfType<mlir::BoolAttr>("asymmetric_quantize_inputs"));
      resultsVec.push_back(result.getResult(0));
    }

    auto results = llvm::ArrayRef<mlir::Value>(resultsVec);
    auto result = rewriter.replaceOpWithNewOp<mlir::TFL::ConcatenationOp>(
        op, outputType, results, 0, "NONE");

    // DEBUG
    llvm::dbgs() << "Result: ";
    result.dump();

    return mlir::success();
  }

#ifdef FULLY_CONNECTED_REDUCE_DIM
  // TODO: see if this can be extended to higher dimensions
  if (inputShape.size() == 3) {
    auto newInputShapeVec = std::vector<int64_t>({
      inputShape[0] * inputShape[1], inputShape[2]});
    auto reshapedInput = Reshape(
        rewriter, op.getLoc(), input, inputType, newInputShapeVec);

    auto weights = op.getOperand(1);
    auto weightsType = weights.getType().dyn_cast<mlir::ShapedType>();
    auto weightsShape = weightsType.getShape();
    auto newFullyConnectedShapeVec = std::vector<int64_t>(newInputShapeVec);
    newFullyConnectedShapeVec.back() = weightsShape.back();
    auto newFullyConnectedShape = llvm::ArrayRef<int64_t>(
        newFullyConnectedShapeVec);
    auto outputType = op.getResult(0).getType().dyn_cast<mlir::ShapedType>();
    auto newFullyConnectedType = mlir::RankedTensorType::get(
        newFullyConnectedShape, outputType.getElementType());
    auto newFullyConnected = rewriter.create<mlir::TFL::FullyConnectedOp>(
        op.getLoc(), newFullyConnectedType, reshapedInput,
        op.getOperand(1), op.getOperand(2),
        op->getAttrOfType<mlir::StringAttr>("fused_activation_function"),
        op->getAttrOfType<mlir::StringAttr>("weights_format"),
        // DEBUG
        // op->getAttrOfType<mlir::BoolAttr>("keep_num_dims")
        rewriter.getBoolAttr(false));

    auto outputShapeLongVec = outputType.getShape().vec();
    auto outputShapeVec = std::vector<int64_t>();
    for (auto dim : outputShapeLongVec)
      outputShapeVec.push_back(static_cast<int64_t>(dim));
    auto output = Reshape(
        rewriter, op.getLoc(), newFullyConnected.getResult(0),
        newFullyConnectedType, outputShapeVec);

    // DEBUG
    llvm::dbgs() << "newFullyConnected: ";
    newFullyConnected.dump();
    llvm::dbgs() << "output: ";
    output.dump();

    rewriter.replaceOp(op, {output});
  } else if (inputShape.size() == 2 && inputShape[0] > 1) {
    // TODO: test using unpack
    auto batchSize = static_cast<int64_t>(inputShape[0]);
    auto numDim = static_cast<int64_t>(2);
    auto stridedSliceInputShape = llvm::ArrayRef<int64_t>(numDim);
    auto stridedSliceInputType = mlir::RankedTensorType::get(
        stridedSliceInputShape, rewriter.getI32Type());
    auto strideAttr = mlir::SplatElementsAttr::get(stridedSliceInputType, 1);
    auto strideConst = rewriter.create<mlir::TFL::ConstOp>(
        op.getLoc(), strideAttr.getType(), strideAttr);

    auto slicedInputShape = inputShape.vec();
    // DEBUG
    // slicedInputShape.erase(slicedInputShape.begin());
    slicedInputShape.front() = 1;
    auto slicedInputType = mlir::RankedTensorType::get(
        slicedInputShape, inputType.getElementType());

    auto zerosMask = rewriter.getI32IntegerAttr(0);
    // DEBUG
    // auto shrinkAxisMask = rewriter.getI32IntegerAttr(1);
    auto shrinkAxisMask = rewriter.getI32IntegerAttr(0);

    auto weights = op.getOperand(1);
    auto weightsType = weights.getType().dyn_cast<mlir::ShapedType>();
    auto weightsShape = weightsType.getShape();
    auto bias = op.getOperand(2);

    // DEBUG
    // auto resultShapeVec = std::vector<int64_t>({weightsShape.back()});
    auto resultShapeVec = std::vector<int64_t>({1, weightsShape.back()});
    auto resultShape = llvm::ArrayRef<int64_t>(resultShapeVec);
    auto outputType = op.getResult(0).getType().dyn_cast<mlir::ShapedType>();
    auto resultType = mlir::RankedTensorType::get(
        resultShape, outputType.getElementType());
    std::vector<mlir::Value> resultsVec;
    resultsVec.reserve(batchSize);
    for (int i = 0; i < batchSize; ++i) {
      // TODO: fix the unpack op's begin and end vals
      auto beginVals = std::vector<int32_t>({i, 0});
      auto beginAttr = rewriter.getI32TensorAttr(beginVals);
      auto beginConst = rewriter.create<mlir::TFL::ConstOp>(
          op.getLoc(), beginAttr.getType(), beginAttr);

      // TODO: check if the vector can be reused
      auto endVals = std::vector<int32_t>(
          {i + 1, static_cast<int32_t>(weightsShape.back())});
      auto endAttr = rewriter.getI32TensorAttr(endVals);
      auto endConst = rewriter.create<mlir::TFL::ConstOp>(
          op.getLoc(), endAttr.getType(), endAttr);

      auto slicedInput = rewriter.create<mlir::TFL::StridedSliceOp>(
          op.getLoc(), slicedInputType, input, beginConst, endConst,
          strideConst, zerosMask, zerosMask, zerosMask, zerosMask,
          shrinkAxisMask);

      // DEBUG
      llvm::dbgs() << "slicedInput: ";
      slicedInput.dump();

      auto result = rewriter.create<mlir::TFL::FullyConnectedOp>(
          op.getLoc(), resultType, slicedInput, weights, bias,
          op->getAttrOfType<mlir::StringAttr>("fused_activation_function"),
          op->getAttrOfType<mlir::StringAttr>("weights_format"),
          // DEBUG
          // op->getAttrOfType<mlir::BoolAttr>("keep_num_dims")
          rewriter.getBoolAttr(false));

      // DEBUG
      llvm::dbgs() << "slicedResult: ";
      result.dump();

      resultsVec.push_back(result.getResult(0));
    }

    auto results = llvm::ArrayRef<mlir::Value>(resultsVec);
    // DEBUG
    /*
    auto result = rewriter.replaceOpWithNewOp<mlir::TFL::PackOp>(
        op, outputType, results,
        static_cast<int32_t>(batchSize), static_cast<int32_t>(0));
    */
    auto result = rewriter.replaceOpWithNewOp<mlir::TFL::ConcatenationOp>(
        op, outputType, results, static_cast<int32_t>(0), "NONE");

    // DEBUG
    llvm::dbgs() << "result: ";
    result.dump();

    return mlir::success();
  }
#endif

  return mlir::failure();
}

ReplaceSplitOp::ReplaceSplitOp(mlir::MLIRContext *ctx)
    : mlir::OpRewritePattern<mlir::TFL::SplitOp>(ctx, /*benefit=*/1) {}

mlir::LogicalResult ReplaceSplitOp::matchAndRewrite(
    mlir::TFL::SplitOp op, mlir::PatternRewriter &rewriter) const {
  llvm::dbgs() << "INFO: SplitOp is called!\n";

  auto splitDim = op.getOperand(0);
  auto splitDimAttr = splitDim.getDefiningOp()->getAttrOfType<
      mlir::ElementsAttr>("value");
  auto splitDimVec = splitDimAttr.getValues<int32_t>();
  auto axis = *splitDimVec.begin();

  // DEBUG
  llvm::dbgs() << "axis: " << axis << "\n";

  if (axis == 0) {
    auto x = op.getOperand(1);
    auto xType = x.getType().dyn_cast<mlir::ShapedType>();

    auto numAttr = op->getAttr("num_splits").dyn_cast<mlir::IntegerAttr>();
    auto num = numAttr.getInt();

    auto size = static_cast<int64_t>(xType.getShape().size());
    auto stridedSliceInputShape = llvm::ArrayRef<int64_t>(size);
    auto stridedSliceInputType = mlir::RankedTensorType::get(
        stridedSliceInputShape, rewriter.getI32Type());
    auto strideAttr = mlir::SplatElementsAttr::get(stridedSliceInputType, 1);
    auto strideConst = rewriter.create<mlir::TFL::ConstOp>(
        op.getLoc(), strideAttr.getType(), strideAttr);

    auto slicedXShape = xType.getShape().vec();
    slicedXShape[axis] = 1;
    auto slicedXType =
        mlir::RankedTensorType::get(slicedXShape, xType.getElementType());

    auto zerosMask = rewriter.getI32IntegerAttr(0);

    std::vector<mlir::Value> slicedXs;
    slicedXs.reserve(num);
    for (long i = 0; i < num; ++i) {
      auto beginVals = std::vector<int32_t>(xType.getShape().size(), 0);
      beginVals[axis] = i;
      auto beginAttr = rewriter.getI32TensorAttr(beginVals);
      auto beginConst = rewriter.create<mlir::TFL::ConstOp>(
          op.getLoc(), beginAttr.getType(), beginAttr);

      // TODO: check if the vector can be reused
      auto endVals = std::vector<int32_t>();
      endVals.reserve(xType.getShape().size());
      for (auto dim : xType.getShape())
        endVals.push_back(static_cast<int32_t>(dim));
      endVals[axis] = i + 1;
      auto endAttr = rewriter.getI32TensorAttr(endVals);
      auto endConst = rewriter.create<mlir::TFL::ConstOp>(
          op.getLoc(), endAttr.getType(), endAttr);

      auto slicedX = rewriter.create<mlir::TFL::StridedSliceOp>(
          op.getLoc(), slicedXType, x, beginConst, endConst, strideConst,
          zerosMask, zerosMask, zerosMask, zerosMask, zerosMask);
      slicedXs.push_back(slicedX);
    }

    rewriter.replaceOp(op, slicedXs);

    return mlir::success();
  }

  return mlir::failure();
}

ReplaceMeanOp::ReplaceMeanOp(mlir::MLIRContext *ctx)
    : mlir::OpRewritePattern<mlir::TFL::MeanOp>(ctx, /*benefit=*/1) {}

mlir::LogicalResult ReplaceMeanOp::matchAndRewrite(
    mlir::TFL::MeanOp op, mlir::PatternRewriter &rewriter) const {
  llvm::dbgs() << "INFO: MeanOp is called!\n";

  auto x = op.getOperand(0);
  auto resultType = op.getResult().getType();
  auto xType = x.getType().dyn_cast<mlir::ShapedType>();
  auto xShapeVec = xType.getShape().vec();
  auto keepDims = op->getAttrOfType<mlir::BoolAttr>(
      "keep_dims").getValue();

  auto axisAttr = op.getOperand(1).getDefiningOp()->getAttr("value");
  auto axisElementsAttr = axisAttr.dyn_cast<mlir::ElementsAttr>();
  auto axis = std::vector<int32_t>();
  for (auto x : axisElementsAttr.getValues<int32_t>())
    axis.push_back(x);
  if (xShapeVec.front() == 1 && axis.size() > 1) {
#ifdef MEAN_SUM_DIV
    auto numElements = 1;
    for (auto dim : xShapeVec)
      numElements *= static_cast<int32_t>(dim);
    auto rDenominator = ScalarConst(rewriter, op.getLoc(), 1.0f / numElements);
    auto sum = rewriter.create<mlir::TFL::SumOp>(
        op.getLoc(), resultType, x, op.getOperand(1), keepDims);
    auto mean = rewriter.replaceOpWithNewOp<mlir::TFL::MulOp>(
        op, resultType, sum, rDenominator, "NONE");
    return mlir::success();
#elif defined(MEAN_SUM_DIV_REPEAT)
    std::sort(axis.begin(), axis.end());

    auto curMean = x;
    auto diff = 0;
    auto reduceDim = false;
    for (auto cur : axis) {
      if (cur == 0) {
        reduceDim = true;
        continue;
      }

      cur -= diff;
      auto curAxisAttr = rewriter.getI32TensorAttr({cur});
      auto curDenom = xShapeVec[cur];
      if (keepDims) {
        xShapeVec[cur] = 1;
      } else {
        xShapeVec.erase(xShapeVec.begin() + cur);
        ++diff;
      }
      auto curShape = mlir::RankedTensorType::get(
          xShapeVec, xType.getElementType());
      auto curAxis = rewriter.create<mlir::TFL::ConstOp>(
          op.getLoc(), curAxisAttr.getType(), curAxisAttr);
      auto curSum = rewriter.create<mlir::TFL::SumOp>(
          op.getLoc(), curShape, curMean, curAxis, keepDims);
      auto curRDenom = ScalarConst(rewriter, op.getLoc(), 1.0f / curDenom);
      curMean = rewriter.create<mlir::TFL::MulOp>(
          op.getLoc(), curShape, curSum, curRDenom, "NONE");
    }

    if (reduceDim && !keepDims) {
      xShapeVec.erase(xShapeVec.begin());
      curMean = Reshape(
          rewriter, op.getLoc(), curMean,
          curMean.getType().dyn_cast<mlir::ShapedType>(), xShapeVec);
    }
    rewriter.replaceOp(op, {curMean});

    return mlir::success();
#elif defined(MEAN_ONE_AXIS)
    std::sort(axis.begin(), axis.end());

    auto curMean = x;
    auto diff = 0;
    auto reduceDim = false;
    for (auto cur : axis) {
      if (cur == 0) {
        reduceDim = true;
        continue;
      }

      cur -= diff;
      auto curAxisAttr = rewriter.getI32TensorAttr({cur});
      if (keepDims) {
        xShapeVec[cur] = 1;
      } else {
        xShapeVec.erase(xShapeVec.begin() + cur);
        ++diff;
      }
      auto curShape = mlir::RankedTensorType::get(
          xShapeVec, xType.getElementType());
      auto curAxis = rewriter.create<mlir::TFL::ConstOp>(
          op.getLoc(), curAxisAttr.getType(), curAxisAttr);
      curMean = rewriter.create<mlir::TFL::MeanOp>(
          op.getLoc(), curShape, curMean, curAxis, keepDims);
    }

    if (reduceDim && !keepDims) {
      xShapeVec.erase(xShapeVec.begin());
      curMean = Reshape(
          rewriter, op.getLoc(), curMean,
          curMean.getType().dyn_cast<mlir::ShapedType>(), xShapeVec);
    }
    rewriter.replaceOp(op, {curMean});

    return mlir::success();
#elif defined(MEAN_RESHAPE)
    // Reshape method only works when the reduce axes are bunched together
    auto newShapeVec = std::vector<int64_t>();
    auto firstAxis = *axisElementsAttr.getValues<int32_t>().begin();
    for (auto i = 0; i < firstAxis; ++i)
      newShapeVec.push_back(xShapeVec[i]);
    auto prevDim = firstAxis - 1;
    auto reduceNumElements = 1;
    auto numAxesReduced = 0;
    for (auto dim : axisElementsAttr.getValues<int32_t>()) {
      // DEBUG
      llvm::dbgs() << "Reduce axis " << dim << "\n";

      if (dim != prevDim + 1)
        return mlir::failure();
      prevDim = dim;
      reduceNumElements *= xShapeVec[dim];
      ++numAxesReduced;
    }
    if (keepDims) {
      for (int i = 1; i < numAxesReduced; ++i)
        newShapeVec.push_back(1);
    }
    newShapeVec.push_back(reduceNumElements);
    for (auto i = prevDim + 1; i < xShapeVec.size(); ++i)
      newShapeVec.push_back(xShapeVec[i]);

    // DEBUG
    llvm::dbgs() << "xType: ";
    xType.dump();
    llvm::dbgs() << "\nnewShapeVec: [";
    for (auto i = 0; i < newShapeVec.size(); ++i) {
      if (i != 0)
        llvm::dbgs() << ", ";
      llvm::dbgs() << newShapeVec[i];
    }
    llvm::dbgs() << "]\n";

    auto reshapedX = Reshape(rewriter, op.getLoc(), x, xType, newShapeVec);

    // DEBUG
    llvm::dbgs() << "Reshaped x: ";
    reshapedX.dump();

    auto newAxisValue = keepDims ? firstAxis + numAxesReduced - 1 : firstAxis;
    auto newAxisAttr = rewriter.getI32TensorAttr({newAxisValue});
    auto newAxis = rewriter.create<mlir::TFL::ConstOp>(
        op.getLoc(), newAxisAttr.getType(), newAxisAttr);

    // DEBUG
    llvm::dbgs() << "New axis: ";
    newAxis.dump();

    auto result = rewriter.replaceOpWithNewOp<mlir::TFL::MeanOp>(
        op, resultType, reshapedX, newAxis, keepDims);

    // DEBUG
    llvm::dbgs() << "Result: ";
    result.dump();

    return mlir::success();
#endif
  }

  return mlir::failure();
}

ReplaceSquaredDifferenceOp::ReplaceSquaredDifferenceOp(mlir::MLIRContext *ctx)
    : mlir::OpRewritePattern<mlir::TFL::SquaredDifferenceOp>(ctx, /*benefit=*/1) {}

mlir::LogicalResult ReplaceSquaredDifferenceOp::matchAndRewrite(
    mlir::TFL::SquaredDifferenceOp op, mlir::PatternRewriter &rewriter) const {
  llvm::dbgs() << "INFO: SquaredDifferenceOp is called!\n";

  auto x = op.getOperand(0);
  auto y = op.getOperand(1);
  auto diff = rewriter.create<mlir::TFL::SubOp>(
      op.getLoc(), x.getType(), x, y, "NONE");
  rewriter.replaceOpWithNewOp<mlir::TFL::MulOp>(
      op, x.getType(), diff, diff, "NONE");

  return mlir::success();
}

ReplaceNegOp::ReplaceNegOp(mlir::MLIRContext *ctx)
    : mlir::OpRewritePattern<mlir::TFL::NegOp>(ctx, /*benefit=*/1) {}

mlir::LogicalResult ReplaceNegOp::matchAndRewrite(
    mlir::TFL::NegOp op, mlir::PatternRewriter &rewriter) const {
  llvm::dbgs() << "INFO: NegOp is called!\n";

  auto x = op.getOperand();
  auto xType = x.getType().dyn_cast<mlir::ShapedType>();

  auto constShape = std::vector<int64_t>();
  auto constType = xType.clone(constShape);
  auto zeroAttr = mlir::SplatElementsAttr::get(constType, 0);
  auto zero = rewriter.create<mlir::TFL::ConstOp>(
      op.getLoc(), zeroAttr.getType(), zeroAttr);

  rewriter.replaceOpWithNewOp<mlir::TFL::SubOp>(
      op, xType, zero, x, "NONE");

  return mlir::success();
}

ReplaceResizeNearestNeighbor::ReplaceResizeNearestNeighbor(mlir::MLIRContext *ctx)
    : mlir::OpRewritePattern<mlir::TFL::ResizeNearestNeighborOp>(ctx, /*benefit=*/1) {}

mlir::LogicalResult ReplaceResizeNearestNeighbor::matchAndRewrite(
    mlir::TFL::ResizeNearestNeighborOp op, mlir::PatternRewriter &rewriter) const {
  llvm::dbgs() << "INFO: ResizeNearestNeighbor is called!\n";

  auto x = op.getOperand(0);
  auto xShape = x.getType().dyn_cast<mlir::ShapedType>().getShape();
  auto resultType = op.getResult().getType().dyn_cast<mlir::ShapedType>();
  auto resultShape = resultType.getShape();
  auto xSize = 1, resultSize = 1;
  bool nonIntegerMultiple = false;
  auto nearestIntegerMultiple = 0;
  for (auto i = 0; i < 4; ++i){
    auto xDim = static_cast<int32_t>(xShape[i]);
    auto resultDim = static_cast<int32_t>(resultShape[i]);
    xSize *= xDim;
    resultSize *= resultDim;
    auto remainder = resultDim % xDim;
    nonIntegerMultiple = nonIntegerMultiple || remainder != 0;
    nearestIntegerMultiple = std::max(
        nearestIntegerMultiple,
        remainder == 0 ? resultDim / xDim : resultDim / xDim + 1);
  }

  // DEBUG
  llvm::dbgs() << "xSize: " << xSize << ", resultSize: " << resultSize
               << ", nonIntegerMultiple: " << nonIntegerMultiple
               << ", nearestIntegerMultiple: " << nearestIntegerMultiple
               << "\n";

  if (resultSize > xSize && nonIntegerMultiple) {
    // Originally, I thought that this operation couldn't be mapped because
    // its tensor size was too big (similar to ResizeBilinear), but it turns
    // out that this operation is unsupported when scaling up by non-integer
    // multiples. Therefore, we scale up by the nearest integer multiple, and
    // then scale down.
    auto upsampledSizeVec = std::vector<int32_t>{
      static_cast<int32_t>(xShape[1]) * nearestIntegerMultiple,
      static_cast<int32_t>(xShape[2]) * nearestIntegerMultiple
    };
    auto upsampledSizeAttr = rewriter.getI32TensorAttr(
        upsampledSizeVec);
    auto upsampledSize = rewriter.create<mlir::TFL::ConstOp>(
        op.getLoc(), upsampledSizeAttr.getType(), upsampledSizeAttr);
    auto upsampledShape = std::vector<int64_t>{
      xShape[0], xShape[1] * nearestIntegerMultiple,
      xShape[2] * nearestIntegerMultiple, xShape[3]
    };
    auto upsampledType = resultType.clone(upsampledShape);
    auto upsampled = rewriter.create<mlir::TFL::ResizeNearestNeighborOp>(
        op.getLoc(), upsampledType, x, upsampledSize,
        op->getAttrOfType<mlir::BoolAttr>("align_corners"),
        op->getAttrOfType<mlir::BoolAttr>("half_pixel_centers"));
    rewriter.replaceOpWithNewOp<mlir::TFL::ResizeNearestNeighborOp>(
        op, resultType, upsampled, op.getOperand(1),
        op->getAttrOfType<mlir::BoolAttr>("align_corners"),
        op->getAttrOfType<mlir::BoolAttr>("half_pixel_centers"));

    return mlir::success();
  } else {
    // GLADNet 240x320
    auto result = ResizePass<mlir::TFL::ResizeNearestNeighborOp>(
        rewriter, op, {1, 96, 96, 64}, {1, 384, 384, 64}, {1, 1, 1, 4});
    if (result.succeeded())
      return result;
    return ResizePass<mlir::TFL::ResizeNearestNeighborOp>(
        rewriter, op, {1, 384, 384, 64}, {1, 240, 320, 64}, {1, 1, 2, 2});
  }
}

ReplaceResizeBilinear::ReplaceResizeBilinear(mlir::MLIRContext *ctx)
    : mlir::OpRewritePattern<mlir::TFL::ResizeBilinearOp>(ctx, /*benefit=*/1) {}

mlir::LogicalResult ReplaceResizeBilinear::matchAndRewrite(
    mlir::TFL::ResizeBilinearOp op, mlir::PatternRewriter &rewriter) const {
  llvm::dbgs() << "INFO: ResizeBilinear is called!\n";

  // U^2 Net 320x320
  auto result = ResizePass<mlir::TFL::ResizeBilinearOp>(
      rewriter, op, {1, 160, 160, 64}, {1, 320, 320, 64}, {1, 2, 2, 1});
  if (result.succeeded())
    return result;

  // U^2 Net 256x256
  result = ResizePass<mlir::TFL::ResizeBilinearOp>(
      rewriter, op, {1, 128, 128, 64}, {1, 256, 256, 64}, {1, 1, 1, 4});
  return result;
}

ReplaceHardSwishOp::ReplaceHardSwishOp(mlir::MLIRContext *ctx)
    : mlir::OpRewritePattern<mlir::TFL::HardSwishOp>(ctx, /*benefit=*/1) {}

mlir::LogicalResult ReplaceHardSwishOp::matchAndRewrite(
    mlir::TFL::HardSwishOp op, mlir::PatternRewriter &rewriter) const {
  llvm::dbgs() << "INFO: HardSwishOp is called!\n";

  auto x = op.getOperand();
  auto xType = x.getType().dyn_cast<mlir::ShapedType>();

  auto constShape = std::vector<int64_t>();
  auto constType = xType.clone(constShape);
  auto threeVal = static_cast<float>(3);
  auto threeAttr = mlir::SplatElementsAttr::get(constType, threeVal);
  auto three = rewriter.create<mlir::TFL::ConstOp>(
      op.getLoc(), threeAttr.getType(), threeAttr);
  auto relu6XPlusThree = rewriter.create<mlir::TFL::AddOp>(
      op.getLoc(), xType, x, three, "RELU6");

  auto y = rewriter.create<mlir::TFL::MulOp>(
      op.getLoc(), xType, x, relu6XPlusThree, "NONE");

  auto oneSixthVal = static_cast<float>(1) / 6;
  auto oneSixthAttr = mlir::SplatElementsAttr::get(constType, oneSixthVal);
  auto oneSixth = rewriter.create<mlir::TFL::ConstOp>(
      op.getLoc(), oneSixthAttr.getType(), oneSixthAttr);
  rewriter.replaceOpWithNewOp<mlir::TFL::MulOp>(
      op, xType, y, oneSixth, "NONE");

  return mlir::success();
}

ReplaceDivOp::ReplaceDivOp(mlir::MLIRContext *ctx)
    : mlir::OpRewritePattern<mlir::TFL::DivOp>(ctx, /*benefit=*/1) {}

mlir::LogicalResult ReplaceDivOp::matchAndRewrite(
    mlir::TFL::DivOp op, mlir::PatternRewriter &rewriter) const {
  llvm::dbgs() << "INFO: DivOp is called!\n";

  auto x = op.getOperand(0);
  auto y = op.getOperand(1);
  auto xType = x.getType().dyn_cast<mlir::ShapedType>();

  auto ySq = rewriter.create<mlir::TFL::MulOp>(
      op.getLoc(), xType, y, y, "NONE");
  auto rAbsY = rewriter.create<mlir::TFL::RsqrtOp>(
      op.getLoc(), xType, ySq);
  auto rAbsY2 = rewriter.create<mlir::TFL::MulOp>(
      op.getLoc(), xType, rAbsY, rAbsY, "NONE");
  auto xy = rewriter.create<mlir::TFL::MulOp>(
      op.getLoc(), xType, x, y, "NONE");
  rewriter.replaceOpWithNewOp<mlir::TFL::MulOp>(op, xType, xy, rAbsY2, "NONE");

  return mlir::success();
}

ReplaceSumOp::ReplaceSumOp(mlir::MLIRContext *ctx)
    : mlir::OpRewritePattern<mlir::TFL::SumOp>(ctx, /*benefit=*/1) {}

mlir::LogicalResult ReplaceSumOp::matchAndRewrite(
    mlir::TFL::SumOp op, mlir::PatternRewriter &rewriter) const {
  llvm::dbgs() << "INFO: SumOp is called!\n";

  auto x = op.getOperand(0);
  auto xType = x.getType().dyn_cast<mlir::ShapedType>();
  auto xShapeVec = xType.getShape().vec();
  auto keepDims = op->getAttrOfType<mlir::BoolAttr>(
      "keep_dims").getValue();

  // DEBUG
  llvm::dbgs() << "xType: ";
  xType.dump();
  llvm::dbgs() << "\n";
  llvm::dbgs() << "result type: ";
  op->getResult(0).getType().dump();
  llvm::dbgs() << "\n";

  auto axisAttr = op.getOperand(1).getDefiningOp()->getAttr("value");
  auto axisElementsAttr = axisAttr.dyn_cast<mlir::ElementsAttr>();
  auto axis = std::vector<int32_t>();
  for (auto x : axisElementsAttr.getValues<int32_t>())
    axis.push_back(x);

  // DEBUG
  llvm::dbgs() << "axisAttr: ";
  axisAttr.dump();
  llvm::dbgs() << "\n";

  if (xShapeVec.front() == 1 && axis.size() > 1) {
    std::sort(axis.begin(), axis.end());

    auto curSum = x;
    auto diff = 0;
    auto reduceDim = false;
    for (auto cur : axis) {
      if (cur == 0) {
        reduceDim = true;
        continue;
      }

      cur -= diff;
      auto curAxisAttr = rewriter.getI32TensorAttr({cur});
      if (keepDims) {
        xShapeVec[cur] = 1;
      } else {
        xShapeVec.erase(xShapeVec.begin() + cur);
        ++diff;
      }
      auto curShape = mlir::RankedTensorType::get(
          xShapeVec, xType.getElementType());
      auto curAxis = rewriter.create<mlir::TFL::ConstOp>(
          op.getLoc(), curAxisAttr.getType(), curAxisAttr);

      // DEBUG
      llvm::dbgs() << "curAxis: ";
      curAxis.dump();

      curSum = rewriter.create<mlir::TFL::SumOp>(
          op.getLoc(), curShape, curSum, curAxis, keepDims);

      // DEBUG
      llvm::dbgs() << "curSum: ";
      curSum.dump();
    }

    if (reduceDim && !keepDims) {
      xShapeVec.erase(xShapeVec.begin());
      curSum = Reshape(
          rewriter, op.getLoc(), curSum,
          curSum.getType().dyn_cast<mlir::ShapedType>(), xShapeVec);

      // DEBUG
      llvm::dbgs() << "reshaped curSum: ";
      curSum.dump();
    }
    rewriter.replaceOp(op, {curSum});


    return mlir::success();
  }

  return mlir::failure();
}

ReplaceExpSum::ReplaceExpSum(
    mlir::MLIRContext *ctx, mlir::PatternBenefit benefit)
    : mlir::RewritePattern(
    mlir::TFL::MulOp::getOperationName(), benefit, ctx) {}

mlir::LogicalResult ReplaceExpSum::match(mlir::Operation *op) const {
  auto activation = op->getAttrOfType<mlir::StringAttr>(
      "fused_activation_function").getValue();
  if (activation != "NONE")
    return mlir::failure();
  auto expOp = op->getOperand(0).getDefiningOp();
  if (!expOp)
    return mlir::failure();
  if (!mlir::isa<mlir::TFL::ExpOp>(expOp))
    return mlir::failure();
  auto powOp = op->getOperand(1).getDefiningOp();
  if (!powOp)
    return mlir::failure();
  if (!mlir::isa<mlir::TFL::PowOp>(powOp))
    return mlir::failure();
  auto sumOp = powOp->getOperand(0).getDefiningOp();
  if (!sumOp)
    return mlir::failure();
  if (!mlir::isa<mlir::TFL::SumOp>(sumOp))
    return mlir::failure();
  auto negOneOp = powOp->getOperand(1).getDefiningOp();
  if (!negOneOp)
    return mlir::failure();
  if (!mlir::isa<mlir::arith::ConstantOp>(negOneOp))
    return mlir::failure();
  auto negOneAttr = negOneOp->getAttrOfType<mlir::ElementsAttr>("value");
  auto negOne = *negOneAttr.getValues<float>().begin();
  if (negOne != -1)
    return mlir::failure();
  if (sumOp->getOperand(0).getDefiningOp() != expOp)
    return mlir::failure();
  return mlir::success();
}

void ReplaceExpSum::rewrite(
    mlir::Operation *op, mlir::PatternRewriter &rewriter) const {
  llvm::dbgs() << "INFO: ReplaceExpSum is called!\n";

  auto expOp = op->getOperand(0).getDefiningOp();
  auto x = expOp->getOperand(0);
  auto resultType = op->getResult(0).getType();
  auto beta = static_cast<float>(1);
  rewriter.replaceOpWithNewOp<mlir::TFL::SoftmaxOp>(
      op, resultType, x, llvm::APFloat(beta));
}

ReplaceMaxPool2DOp::ReplaceMaxPool2DOp(mlir::MLIRContext *ctx)
    : mlir::OpRewritePattern<mlir::TFL::MaxPool2DOp>(ctx) {}

mlir::LogicalResult ReplaceMaxPool2DOp::matchAndRewrite(
    mlir::TFL::MaxPool2DOp op, mlir::PatternRewriter &rewriter) const {
  llvm::dbgs() << "INFO: MaxPool2DOp is called!\n";

  auto x = op.getOperand();
  auto resultType = op->getResult(0).getType();
  auto maxpool = rewriter.create<mlir::TFL::MaxPool2DOp>(
      op.getLoc(), resultType, x,
      op->getAttrOfType<mlir::StringAttr>("padding"),
      op->getAttrOfType<mlir::IntegerAttr>("stride_w"),
      op->getAttrOfType<mlir::IntegerAttr>("stride_h"),
      op->getAttrOfType<mlir::IntegerAttr>("filter_width"),
      op->getAttrOfType<mlir::IntegerAttr>("filter_height"),
      rewriter.getStringAttr("NONE"));
  auto activation = op->getAttrOfType<mlir::StringAttr>(
      "fused_activation_function").getValue();
  if (activation == "RELU") {
    rewriter.replaceOpWithNewOp<mlir::TFL::ReluOp>(op, resultType, maxpool);
    return mlir::success();
  }
  return mlir::failure();
}

ReplaceAveragePool2DOp::ReplaceAveragePool2DOp(mlir::MLIRContext *ctx)
    : mlir::OpRewritePattern<mlir::TFL::AveragePool2DOp>(ctx) {}

mlir::LogicalResult ReplaceAveragePool2DOp::matchAndRewrite(
    mlir::TFL::AveragePool2DOp op, mlir::PatternRewriter &rewriter) const {
  llvm::dbgs() << "INFO: AveragePool2DOp is called!\n";

  auto x = op.getOperand();
  auto resultType = op->getResult(0).getType();

  auto xType = x.getType().dyn_cast<mlir::ShapedType>();
  auto xShapeVec = xType.getShape().vec();
  auto filterW = op->getAttrOfType<mlir::IntegerAttr>(
      "filter_width").getInt();
  auto filterH = op->getAttrOfType<mlir::IntegerAttr>(
      "filter_height").getInt();
  auto xNumElements = 1;
  for (auto dimSize : xShapeVec)
    xNumElements *= dimSize;

  // DEBUG
  llvm::dbgs() << "Input width: " << xShapeVec[2] << "\n";
  llvm::dbgs() << "Input height: " << xShapeVec[1] << "\n";
  llvm::dbgs() << "Filter width: " << filterW << "\n";
  llvm::dbgs() << "Input height: " << filterH << "\n";
  llvm::dbgs() << "Input num elements: " << xNumElements << "\n";

  if (filterH == xShapeVec[1] && filterW == xShapeVec[2] &&
      xNumElements >= 147456) {
    auto axisAttr = rewriter.getI32TensorAttr({1, 2});
    auto axis = rewriter.create<mlir::TFL::ConstOp>(
        op.getLoc(), axisAttr.getType(), axisAttr);
    auto mean = rewriter.create<mlir::TFL::MeanOp>(
        op.getLoc(), resultType, x, axis, true);

    auto activation = op->getAttrOfType<mlir::StringAttr>(
        "fused_activation_function").getValue();
    if (activation == "RELU") {
      rewriter.replaceOpWithNewOp<mlir::TFL::ReluOp>(op, resultType, mean);
      return mlir::success();
    } else if (activation == "NONE") {
      rewriter.replaceOp(op, {mean});
      return mlir::success();
    }
    return mlir::failure();
  }
  return mlir::failure();
}

ReplaceReshapeOp::ReplaceReshapeOp(mlir::MLIRContext *ctx)
    : mlir::OpRewritePattern<mlir::TFL::ReshapeOp>(ctx) {}

mlir::LogicalResult ReplaceReshapeOp::matchAndRewrite(
    mlir::TFL::ReshapeOp op, mlir::PatternRewriter &rewriter) const {
  llvm::dbgs() << "INFO: ReshapeOp is called!\n";

  auto x = op.getOperand(0);
  auto xType = x.getType().dyn_cast<mlir::ShapedType>();
  auto xShapeVec = xType.getShape().vec();
  auto resultType = op->getResult(0).getType().dyn_cast<mlir::ShapedType>();
  auto resultShapeVec = resultType.getShape().vec();

  auto xNumElements = 1;
  for (auto dimSize : xShapeVec)
    xNumElements *= dimSize;
  // This split and reshape only works when multiple axes are reshaped into one
  if (xNumElements >= 73728 &&
      xShapeVec.size() == 4 &&
      resultShapeVec.size() == 2 &&
      resultShapeVec[0] == xShapeVec[1] * xShapeVec[2] &&
      resultShapeVec[1] == xShapeVec[3]) {
    auto splits = std::vector<mlir::Value>();
    auto numSplits = 2;
    auto splitSize = static_cast<int32_t>(xShapeVec[1] / numSplits);
    for (auto i = 0; i < numSplits; ++i) {
      auto split = StridedSlice(
          rewriter, op.getLoc(), x, xType,
          {
              xShapeVec[0],
              splitSize,
              xShapeVec[2],
              xShapeVec[3]
          },
          {0, i * splitSize, 0, 0},
          {
              static_cast<int32_t>(xShapeVec[0]),
              (i + 1) * splitSize,
              static_cast<int32_t>(xShapeVec[2]),
              static_cast<int32_t>(xShapeVec[3])
          },
          {1, 1, 1, 1}
      );
      auto reshapedSplit = Reshape(
          rewriter, op.getLoc(), split,
          split.getType().dyn_cast<mlir::ShapedType>(),
          {
              resultShapeVec[0] / numSplits,
              static_cast<int32_t>(resultShapeVec[1])
          });
      splits.push_back(reshapedSplit);
    }

    auto result = Concat(rewriter, op.getLoc(), splits, 0);
    rewriter.replaceOp(op, {result});

    return mlir::success();
  }
  return mlir::failure();
}

ReplaceMulPow::ReplaceMulPow(mlir::MLIRContext *ctx,
                             mlir::PatternBenefit benefit)
    : mlir::RewritePattern(mlir::TFL::MulOp::getOperationName(), benefit,
                           ctx) {}

mlir::LogicalResult ReplaceMulPow::match(mlir::Operation *op) const {
  auto activation = op->getAttrOfType<mlir::StringAttr>(
      "fused_activation_function").getValue();
  if (activation != "NONE")
    return mlir::failure();
  auto x = op->getOperand(0).getDefiningOp();
  auto y = op->getOperand(1).getDefiningOp();
  auto xIsPow = x && mlir::isa<mlir::TFL::PowOp>(x);
  auto yIsPow = y && mlir::isa<mlir::TFL::PowOp>(y);
  if (xIsPow || yIsPow) {
    if (xIsPow && yIsPow)
      return mlir::failure();
    if (xIsPow)
      std::swap(x, y);
  } else {
    return mlir::failure();
  }
  auto pOp = y->getOperand(1).getDefiningOp();
  if (!mlir::isa<mlir::arith::ConstantOp>(pOp))
    return mlir::failure();
  auto pAttr = pOp->getAttrOfType<mlir::ElementsAttr>("value");
  auto pValue = *pAttr.getValues<float>().begin();
  if (pValue != -1)
    return mlir::failure();
  return mlir::success();
}

void ReplaceMulPow::rewrite(mlir::Operation *op,
                            mlir::PatternRewriter &rewriter) const {
  llvm::dbgs() << "INFO: ReplaceMulPow is called!\n";

  auto x = op->getOperand(0);
  auto y = op->getOperand(1);
  auto xOp = x.getDefiningOp();
  if (xOp && mlir::isa<mlir::TFL::PowOp>(xOp))
    std::swap(x, y);
  y = y.getDefiningOp()->getOperand(0);
  auto result = rewriter.replaceOpWithNewOp<mlir::TFL::DivOp>(
      op, op->getResult(0).getType(), x, y, "NONE");

  // DEBUG
  llvm::dbgs() << "result: ";
  result.dump();
}

ReplacePowOp::ReplacePowOp(mlir::MLIRContext *ctx)
  : mlir::OpRewritePattern<mlir::TFL::PowOp>(ctx) {}

mlir::LogicalResult ReplacePowOp::matchAndRewrite(
    mlir::TFL::PowOp op, mlir::PatternRewriter &rewriter) const {
  llvm::dbgs() << "INFO: ReplacePowOp is called!\n";

  auto x = op.getOperand(0);
  auto p = op.getOperand(1);
  auto pOp = p.getDefiningOp();
  if (!pOp || !mlir::isa<mlir::arith::ConstantOp>(pOp))
    return mlir::failure();
  auto pAttr = pOp->getAttrOfType<mlir::ElementsAttr>("value");
  auto pValue = *pAttr.getValues<float>().begin();
  if (pValue != -0.5)
    return mlir::failure();
  rewriter.replaceOpWithNewOp<mlir::TFL::RsqrtOp>(
      op, op.getResult().getType(), x);
  return mlir::success();
}

ReplaceMirrorPadOp::ReplaceMirrorPadOp(mlir::MLIRContext *ctx)
    : mlir::OpRewritePattern<mlir::TFL::MirrorPadOp>(ctx) {}

mlir::LogicalResult ReplaceMirrorPadOp::matchAndRewrite(
    mlir::TFL::MirrorPadOp op, mlir::PatternRewriter &rewriter) const {
  llvm::dbgs() << "INFO: MirrorPadOp is called!\n";

  auto x = op.getOperand(0);
  auto xType = x.getType().dyn_cast<mlir::ShapedType>();
  auto xShapeVec = xType.getShape().vec();

  // DEBUG
  llvm::dbgs() << "xShapeVec: ";
  for (auto dim : xShapeVec)
    llvm::dbgs() << dim << " ";
  llvm::dbgs() << "\n";

  // We only support NHWC
  if (xShapeVec.size() != 4)
    return mlir::failure();
  auto padding = op.getOperand(1);

  // DEBUG
  llvm::dbgs() << "padding:\n";
  padding.dump();

  auto paddingOp = padding.getDefiningOp();
  if (!paddingOp || !mlir::isa<mlir::arith::ConstantOp>(paddingOp))
    return mlir::failure();
  auto paddingAttr = paddingOp->getAttrOfType<mlir::ElementsAttr>(
      "value").getValues<int32_t>();

  // DEBUG
  llvm::dbgs() << "paddingAttr: ";
  for (auto val : paddingAttr)
    llvm::dbgs() << val << " ";
  llvm::dbgs() << "\n";

  // We only support padding along the H and W axis
  if (paddingAttr[0] != 0 || paddingAttr[1] != 0 ||
      paddingAttr[6] != 0 || paddingAttr[7] != 0)
    return mlir::failure();

  auto n = xShapeVec[0], h = xShapeVec[1], w = xShapeVec[2], c = xShapeVec[3];
  auto leftPad = paddingAttr[4];
  auto rightPad = paddingAttr[5];
  auto xs = std::vector<mlir::Value>{};

  // DEBUG
  llvm::dbgs() << "leftPad (" << leftPad << "):\n";

  for (auto i = leftPad; i > 0; --i) {
    xs.push_back(StridedSlice(
        rewriter, op.getLoc(), x, xType,
        {n, h, 1, c},
        {0, 0, i, 0},
        {static_cast<int32_t>(n), static_cast<int32_t>(h), i + 1,
         static_cast<int32_t>(c)},
        {1, 1, 1, 1}));

    // DEBUG
    xs.back().dump();
  }
  xs.push_back(x);

  // DEBUG
  llvm::dbgs() << "rightPad (" << rightPad << "):\n";

  for (auto i = static_cast<int32_t>(w) - 2; i >= w - rightPad - 1; --i) {
    xs.push_back(StridedSlice(
        rewriter, op.getLoc(), x, xType,
        {n, h, 1, c},
        {0, 0, i, 0},
        {static_cast<int32_t>(n), static_cast<int32_t>(h), i + 1,
         static_cast<int32_t>(c)},
        {1, 1, 1, 1}));

    // DEBUG
    xs.back().dump();
  }
  x = Concat(rewriter, op.getLoc(), xs, 2);

  // DEBUG
  llvm::dbgs() << "Concat left and right:\n";
  x.dump();

  w += leftPad + rightPad;

  auto topPad = paddingAttr[2];
  auto bottomPad = paddingAttr[3];
  xs = std::vector<mlir::Value>{};

  // DEBUG
  llvm::dbgs() << "topPad (" << topPad << "):\n";

  for (auto i = topPad; i > 0; --i) {
    xs.push_back(StridedSlice(
        rewriter, op.getLoc(), x, xType,
        {n, 1, w, c},
        {0, i, 0, 0},
        {static_cast<int32_t>(n), i + 1, static_cast<int32_t>(w),
         static_cast<int32_t>(c)},
        {1, 1, 1, 1}));

    // DEBUG
    xs.back().dump();
  }
  xs.push_back(x);

  // DEBUG
  llvm::dbgs() << "bottomPad (" << bottomPad << "):\n";

  for (auto i = static_cast<int32_t>(h) - 2; i >= h - bottomPad - 1; --i) {
    xs.push_back(StridedSlice(
        rewriter, op.getLoc(), x, xType,
        {n, 1, w, c},
        {0, i, 0, 0},
        {static_cast<int32_t>(n), i + 1, static_cast<int32_t>(w),
         static_cast<int32_t>(c)},
        {1, 1, 1, 1}));

    // DEBUG
    xs.back().dump();
  }
  x = Concat(rewriter, op.getLoc(), xs, 1);

  // DEBUG
  llvm::dbgs() << "Concat top and bottom:\n";
  x.dump();

  rewriter.replaceOp(op, {x});

  return mlir::success();
}

ReplaceSquareOp::ReplaceSquareOp(mlir::MLIRContext *ctx)
    : mlir::OpRewritePattern<mlir::TFL::SquareOp>(ctx) {}

mlir::LogicalResult ReplaceSquareOp::matchAndRewrite(
    mlir::TFL::SquareOp op, mlir::PatternRewriter &rewriter) const {
  llvm::dbgs() << "INFO: SquareOp is called!\n";

  auto x = op.getOperand();
  rewriter.replaceOpWithNewOp<mlir::TFL::MulOp>(
      op, op.getResult().getType(), x, x, "NONE");

  return mlir::success();
}

ReplaceSoftplusTanhMul::ReplaceSoftplusTanhMul(
    mlir::MLIRContext *ctx, const std::string &savedModelPath,
    mlir::PatternBenefit benefit)
    : mlir::RewritePattern(mlir::TFL::MulOp::getOperationName(),
                           benefit, ctx),
      calibrationData_(savedModelPath, "Mul_approx.txt") {}

mlir::LogicalResult ReplaceSoftplusTanhMul::match(mlir::Operation *op) const {
  auto activation = op->getAttrOfType<mlir::StringAttr>(
      "fused_activation_function");
  if (activation != "NONE")
    return mlir::failure();
  auto tanhOp = op->getOperand(1).getDefiningOp();
  if (!tanhOp || !mlir::isa<mlir::TFL::TanhOp>(tanhOp))
    return mlir::failure();
  auto conv2DOp = op->getOperand(0).getDefiningOp();
  if (!conv2DOp || !mlir::isa<mlir::TFL::Conv2DOp>(conv2DOp))
    return mlir::failure();
  activation = conv2DOp->getAttrOfType<mlir::StringAttr>(
      "fused_activation_function");
  if (activation != "NONE")
    return mlir::failure();
  auto logOp = tanhOp->getOperand(0).getDefiningOp();
  if (!logOp || !mlir::isa<mlir::TFL::LogOp>(logOp))
    return mlir::failure();
  auto addOp = logOp->getOperand(0).getDefiningOp();
  if (!addOp || !mlir::isa<mlir::TFL::AddOp>(addOp))
    return mlir::failure();
  activation = addOp->getAttrOfType<mlir::StringAttr>(
      "fused_activation_function");
  if (activation != "NONE")
    return mlir::failure();
  auto oneOp = addOp->getOperand(1).getDefiningOp();
  if (!oneOp || !mlir::isa<mlir::arith::ConstantOp>(oneOp))
    return mlir::failure();
  auto oneAttr = oneOp->getAttrOfType<mlir::ElementsAttr>("value");
  auto one = *oneAttr.getValues<float>().begin();
  if (one != 1)
    return mlir::failure();
  auto expOp = addOp->getOperand(0).getDefiningOp();
  if (!expOp || !mlir::isa<mlir::TFL::ExpOp>(expOp))
    return mlir::failure();
  auto otherConv2DOp = expOp->getOperand(0).getDefiningOp();
  if (otherConv2DOp != conv2DOp)
    return mlir::failure();
  return mlir::success();
}

void ReplaceSoftplusTanhMul::rewrite(mlir::Operation *op,
                                     mlir::PatternRewriter &rewriter) const {
  llvm::dbgs() << "INFO: SoftplusTanhMul is called!\n";
#ifdef SOFTPLUS_TANH_MUL_RELU
  auto conv2DOp = op->getOperand(0).getDefiningOp();
  rewriter.replaceOpWithNewOp<mlir::TFL::Conv2DOp>(
      op, conv2DOp->getResult(0).getType(), conv2DOp->getOperand(0),
      conv2DOp->getOperand(1), conv2DOp->getOperand(2),
      conv2DOp->getAttrOfType<mlir::IntegerAttr>("dilation_h_factor"),
      conv2DOp->getAttrOfType<mlir::IntegerAttr>("dilation_w_factor"),
      rewriter.getStringAttr("RELU"),
      conv2DOp->getAttrOfType<mlir::StringAttr>("padding"),
      conv2DOp->getAttrOfType<mlir::IntegerAttr>("stride_h"),
      conv2DOp->getAttrOfType<mlir::IntegerAttr>("stride_w"));
#elif defined(SOFTPLUS_TANH_MUL_SIGMOID_MUL) || \
      defined(SOFTPLUS_TANH_MUL_SIGMOID_NO_MUL)
  auto x = op->getOperand(0);
  auto xType = x.getType();
  auto scale = ScalarConst(rewriter, op->getLoc(), 1.3);
  auto offset = ScalarConst(rewriter, op->getLoc(), 0.311896);
  auto offsetX = rewriter.create<mlir::TFL::AddOp>(
      op->getLoc(), xType, x, offset, "NONE");
  auto scaledX = rewriter.create<mlir::TFL::MulOp>(
      op->getLoc(), xType, scale, offsetX, "NONE");
  auto sigmoidX = rewriter.create<mlir::TFL::LogisticOp>(
      op->getLoc(), xType, scaledX);
#ifdef SOFTPLUS_TANH_MUL_SIGMOID_MUL
  rewriter.replaceOpWithNewOp<mlir::TFL::MulOp>(op, xType, x, sigmoidX,
                                                "NONE");
#else
  auto xSq = rewriter.create<mlir::TFL::SquareOp>(op->getLoc(), xType, x);
  auto sigmoidXSq = rewriter.create<mlir::TFL::SquareOp>(
      op->getLoc(), xType, sigmoidX);
  auto squareSum = rewriter.create<mlir::TFL::AddOp>(
      op->getLoc(), xType, xSq, sigmoidXSq, "NONE");
  auto squaredDiff = rewriter.create<mlir::TFL::SquaredDifferenceOp>(
      op->getLoc(), xType, x, sigmoidX);
  auto twiceResult = rewriter.create<mlir::TFL::SubOp>(
      op->getLoc(), xType, squareSum, squaredDiff, "NONE");
  auto half = ScalarConst(rewriter, op->getLoc(), 0.5);
  rewriter.replaceOpWithNewOp<mlir::TFL::MulOp>(
      op, xType, half, twiceResult, "NONE");
#endif
#elif defined(SOFTPLUS_TANH_MUL_LINE_SEGMENTS)
  auto x = op->getOperand(0);
  auto coeffs = calibrationData_.getCoeffs(op->getLoc());
  auto result = PiecewiseRegression(rewriter, op->getLoc(), x, coeffs);
  rewriter.replaceOp(op, {result});
#endif
}

AllTFLPasses::AllTFLPasses(const std::string &savedModelPath)
    : savedModelPath_(savedModelPath) {}

void AllTFLPasses::runOnOperation() {
  auto ctx = &getContext();
  mlir::RewritePatternSet patterns(ctx);
  patterns.insert(std::make_unique<ReplaceAbsOp>(ctx));
  patterns.insert(std::make_unique<ReplaceLeakyReluOp>(ctx));
  patterns.insert(std::make_unique<ReplaceTileOp>(ctx));
#ifdef UNPACK_TFL
  patterns.insert(std::make_unique<ReplaceTFLUnpackOp>(ctx));
#endif
#ifndef EXP_DISABLE
  patterns.insert(std::make_unique<ReplaceExpOp>(ctx));
#endif
#ifndef LOG_DISABLE
  patterns.insert(std::make_unique<ReplaceLogOp>(ctx));
#endif
  patterns.insert(std::make_unique<ReplaceFullyConnectedOp>(ctx));
  patterns.insert(std::make_unique<ReplaceSplitOp>(ctx));
  patterns.insert(std::make_unique<ReplaceMeanOp>(ctx));
#ifdef SQUARED_DIFFERENCE_OLD
  patterns.insert(std::make_unique<ReplaceSquaredDifferenceOp>(ctx));
#endif
  patterns.insert(std::make_unique<ReplaceNegOp>(ctx));
  patterns.insert(std::make_unique<ReplaceResizeNearestNeighbor>(ctx));
  patterns.insert(std::make_unique<ReplaceResizeBilinear>(ctx));
  patterns.insert(std::make_unique<ReplaceHardSwishOp>(ctx));
  patterns.insert(std::make_unique<ReplaceDivOp>(ctx));
  patterns.insert(std::make_unique<ReplaceSumOp>(ctx));
  patterns.insert(std::make_unique<ReplaceExpSum>(ctx));
  patterns.insert(std::make_unique<ReplaceMaxPool2DOp>(ctx));
  patterns.insert(std::make_unique<ReplaceAveragePool2DOp>(ctx));
  patterns.insert(std::make_unique<ReplaceReshapeOp>(ctx));
  patterns.insert(std::make_unique<ReplaceMulPow>(ctx));
  patterns.insert(std::make_unique<ReplacePowOp>(ctx));
  patterns.insert(std::make_unique<ReplaceMirrorPadOp>(ctx));
  patterns.insert(std::make_unique<ReplaceSquareOp>(ctx));
#if defined(SOFTPLUS_TANH_MUL_RELU) || \
    defined(SOFTPLUS_TANH_MUL_SIGMOID_MUL) || \
    defined(SOFTPLUS_TANH_MUL_SIGMOID_NO_MUL) || \
    defined(SOFTPLUS_TANH_MUL_LINE_SEGMENTS)
  patterns.insert(std::make_unique<ReplaceSoftplusTanhMul>(
      ctx, savedModelPath_));
#endif

  mlir::applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
}

ReplaceSoftplusOp::ReplaceSoftplusOp(mlir::MLIRContext *ctx,
                                     const std::string &savedModelPath)
    : mlir::OpRewritePattern<mlir::TF::SoftplusOp>(ctx, /*benefit=*/1),
      calibrationData_(savedModelPath, "Softplus_approx.txt") {}

mlir::LogicalResult ReplaceSoftplusOp::matchAndRewrite(
    mlir::TF::SoftplusOp op, mlir::PatternRewriter &rewriter) const {
  llvm::dbgs() << "INFO: SoftplusOp is called!\n";

#ifdef SOFTPLUS_X
  rewriter.replaceOp(op, {op.getOperand()});
#endif

#ifdef SOFTPLUS_RELU
  rewriter.replaceOpWithNewOp<mlir::TF::ReluOp>(op, op.getOperand());
#endif

#ifdef SOFTPLUS_LINE_SEGMENTS
  auto coeffs = calibrationData_.getCoeffs(op.getLoc());
  auto result = PiecewiseRegression(
      rewriter, op.getLoc(), op.getOperand(), coeffs);
  rewriter.replaceOp(op, {result});
#endif

#if defined(SOFTPLUS_LEAST_SQ) || defined(SOFTPLUS_REL_LEAST_SQ) || \
    defined(SOFTPLUS_MINIMAX) || defined(SOFTPLUS_TAYLOR)
  auto x = op.getOperand();
  float coeffs[20];

#ifdef SOFTPLUS_LEAST_SQ
  int degree = 3;
  calibrationData_.getPolyCoeff(
      op.getLoc(), "softplus", "least_square", degree, coeffs);
#endif
#ifdef SOFTPLUS_REL_LEAST_SQ
  int degree = 3;
  calibrationData_.getPolyCoeff(
      op.getLoc(), "softplus", "relative_least_square", degree, coeffs);
#endif
#ifdef SOFTPLUS_MINIMAX
  int degree = 3;
  calibrationData_.getPolyCoeff(
      op.getLoc(), "softplus", "minimax", degree, coeffs);
#endif
#ifdef SOFTPLUS_TAYLOR
  int degree = 3;
  calibrationData_.getPolyCoeff(
      op.getLoc(), "softplus", "taylor", degree, coeffs);
#endif
  auto sum = PolynomialValueTF(rewriter, op.getLoc(), x, coeffs, degree);
  rewriter.replaceOp(op, {sum});
#endif

  return mlir::success();
}

ReplaceTFUnpackOp::ReplaceTFUnpackOp(mlir::MLIRContext *ctx)
    : mlir::OpRewritePattern<mlir::TF::UnpackOp>(ctx, /*benefit=*/1) {}

mlir::LogicalResult ReplaceTFUnpackOp::matchAndRewrite(
    mlir::TF::UnpackOp op, mlir::PatternRewriter &rewriter) const {
  llvm::dbgs() << "INFO: UnpackOp is called!\n";

  auto x = op.getOperand();
  auto xType = x.getType().dyn_cast<mlir::ShapedType>();
  auto axisAttr = op->getAttr("axis").dyn_cast<mlir::IntegerAttr>();
  auto axis = axisAttr.getInt();
  auto num = xType.getShape()[axis];

  // DEBUG
  llvm::dbgs() << "num = " << num << ", axis = " << axis << "\n";

  auto size = static_cast<int64_t>(xType.getShape().size());
  auto stridedSliceInputShape = llvm::ArrayRef<int64_t>(size);
  auto stridedSliceInputType = mlir::RankedTensorType::get(
      stridedSliceInputShape, rewriter.getI32Type());
  auto strideAttr = mlir::SplatElementsAttr::get(stridedSliceInputType, 1);
  auto strideConst = rewriter.create<mlir::TF::ConstOp>(
      op.getLoc(), strideAttr.getType(), strideAttr);

  // DEBUG
  llvm::dbgs() << "strides: ";
  strideConst.dump();
  llvm::dbgs() << "\n";

  auto slicedXShape = xType.getShape().vec();
  slicedXShape.erase(slicedXShape.begin() + axis);
  auto slicedXType =
      mlir::RankedTensorType::get(slicedXShape, xType.getElementType());

  auto ui64Type = mlir::IntegerType::get(rewriter.getContext(), 64);
  auto zerosMask = rewriter.getIntegerAttr(ui64Type, 0);
  auto shrinkAxisMask = rewriter.getIntegerAttr(ui64Type, 1 << axis);

  std::vector<mlir::Value> slicedXs;
  slicedXs.reserve(num);
  for (long i = 0; i < num; ++i) {
    auto beginVals = std::vector<int32_t>(xType.getShape().size(), 0);
    beginVals[axis] = i;
    auto beginAttr = rewriter.getI32TensorAttr(beginVals);
    auto beginConst = rewriter.create<mlir::TF::ConstOp>(
        op.getLoc(), beginAttr.getType(), beginAttr);

    // DEBUG
    llvm::dbgs() << "begin: ";
    beginConst.dump();
    llvm::dbgs() << "\n";

    // TODO: check if the vector can be reused
    auto endVals = std::vector<int32_t>();
    endVals.reserve(xType.getShape().size());
    for (auto dim : xType.getShape())
      endVals.push_back(static_cast<int32_t>(dim));
    endVals[axis] = i + 1;
    auto endAttr = rewriter.getI32TensorAttr(endVals);
    auto endConst = rewriter.create<mlir::TF::ConstOp>(
        op.getLoc(), endAttr.getType(), endAttr);

    // DEBUG
    llvm::dbgs() << "end: ";
    endConst.dump();
    llvm::dbgs() << "\n";

    auto slicedX = rewriter.create<mlir::TF::StridedSliceOp>(
        op.getLoc(), slicedXType, x, beginConst, endConst, strideConst,
        zerosMask, zerosMask, zerosMask, zerosMask, shrinkAxisMask);

    // DEBUG
    llvm::dbgs() << "strided slice: ";
    slicedX.dump();
    llvm::dbgs() << "\n";

    slicedXs.push_back(slicedX);
  }

  rewriter.replaceOp(op, slicedXs);

  return mlir::success();
}

#define MY_TF_PASS
AllTFPasses::AllTFPasses(const std::string &savedModelPath)
    : savedModelPath_(savedModelPath) {}

void AllTFPasses::runOnOperation() {
  auto ctx = &getContext();
  mlir::RewritePatternSet patterns(ctx);
#if defined(SOFTPLUS_RELU) || defined(SOFTPLUS_X) || \
    defined(SOFTPLUS_LINE_SEGMENTS) || defined(SOFTPLUS_LEAST_SQ) || \
    defined(SOFTPLUS_REL_LEAST_SQ) || defined(SOFTPLUS_MINIMAX) ||   \
    defined(SOFTPLUS_TAYLOR)
  patterns.insert(std::make_unique<ReplaceSoftplusOp>(
      ctx, savedModelPath_));
#endif
#ifdef UNPACK_TF
  patterns.insert(std::make_unique<ReplaceTFUnpackOp>(ctx));
#endif
  mlir::applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
}

mlir::Value PolynomialValueTF(
    mlir::PatternRewriter &rewriter, const mlir::Location &loc,
    const mlir::Value &x, float coeffs[], int degree) {
  auto xType = x.getType().dyn_cast<mlir::ShapedType>();

  auto constShape = std::vector<int64_t>();
  // DEBUG
  // auto constType = xType;
  auto constType = xType.clone(constShape);

  auto highestTermCoeffAttr = mlir::SplatElementsAttr::get(
      constType, coeffs[degree]);
  auto highestTermCoeff = rewriter.create<mlir::TF::ConstOp>(
      loc, highestTermCoeffAttr.getType(), highestTermCoeffAttr);

  auto sum = highestTermCoeff.getResult();
  for (int i = degree - 1; i >= 0; --i) {
    sum = rewriter.create<mlir::TF::MulOp>(loc, sum, x);
    auto coeffAttr = mlir::SplatElementsAttr::get(constType, coeffs[i]);
    auto coeff = rewriter.create<mlir::TF::ConstOp>(
        loc, coeffAttr.getType(), coeffAttr);
    sum = rewriter.create<mlir::TF::AddOp>(loc, sum, coeff);
  }

  return sum;
}

mlir::Value PolynomialValueTFL(
    mlir::PatternRewriter &rewriter, const mlir::Location &loc,
    const mlir::Value &x, float coeffs[], int degree) {
  auto xType = x.getType().dyn_cast<mlir::ShapedType>();

  auto constShape = std::vector<int64_t>();
  // DEBUG
  // auto constType = xType;
  auto constType = xType.clone(constShape);

  auto highestTermCoeffAttr = mlir::SplatElementsAttr::get(
      constType, coeffs[degree]);
  auto highestTermCoeff = rewriter.create<mlir::TFL::ConstOp>(
      loc, highestTermCoeffAttr.getType(), highestTermCoeffAttr);

  auto sum = highestTermCoeff.getResult();
  auto noActivationAttr = rewriter.getStringAttr("NONE");
  for (int i = degree - 1; i >= 0; --i) {
    sum = rewriter.create<mlir::TFL::MulOp>(loc, sum, x, noActivationAttr);
    auto coeffAttr = mlir::SplatElementsAttr::get(constType, coeffs[i]);
    auto coeff = rewriter.create<mlir::TFL::ConstOp>(
        loc, coeffAttr.getType(), coeffAttr);
    sum = rewriter.create<mlir::TFL::AddOp>(
        loc, sum, coeff, noActivationAttr);
  }

  return sum;
}

mlir::Value Reshape(
    mlir::PatternRewriter &rewriter, const mlir::Location &loc,
    const mlir::Value &x, const mlir::ShapedType &xType,
    const std::vector<int64_t> &newXShapeVec) {
  auto newXShape = llvm::ArrayRef<int64_t>(newXShapeVec);
  auto newXShapeI32Vec = std::vector<int32_t>();
  for (auto dim : newXShape)
    newXShapeI32Vec.push_back(static_cast<int32_t>(dim));
  auto newXShapeI32 = llvm::ArrayRef<int32_t>(newXShapeI32Vec);
  auto newXType = mlir::RankedTensorType::get(
      newXShape, xType.getElementType());
  auto newXShapeRank = static_cast<int64_t>(newXShape.size());
  auto newXShapeShape = llvm::ArrayRef<int64_t>(newXShapeRank);
  auto newXShapeType = mlir::RankedTensorType::get(
      newXShapeShape, rewriter.getI32Type());
  auto newXShapeAttr = mlir::DenseElementsAttr::get(
      newXShapeType, newXShapeI32);
  auto newXShapeConst = rewriter.create<mlir::TFL::ConstOp>(
      loc, newXShapeAttr.getType(), newXShapeAttr);
  auto reshapedX = rewriter.create<mlir::TFL::ReshapeOp>(
      loc, newXType, x, newXShapeConst);
  return reshapedX;
}

mlir::Value StridedSlice(
    mlir::PatternRewriter &rewriter, const mlir::Location &loc,
    const mlir::Value &x, mlir::ShapedType xType,
    const std::vector<int64_t> &newXShapeVec,
    const std::vector<int32_t> &beginVec,
    const std::vector<int32_t> &endVec,
    const std::vector<int32_t> &stridesVec,
    int64_t beginMask,
    int64_t endMask,
    int64_t ellipsisMask,
    int64_t newAxisMask,
    int64_t shrinkAxisMask) {
  auto slicedType = xType.clone(newXShapeVec);

  auto beginAttr = rewriter.getI32TensorAttr(beginVec);
  auto beginConst = rewriter.create<mlir::TF::ConstOp>(
      loc, beginAttr.getType(), beginAttr);

  auto endAttr = rewriter.getI32TensorAttr(endVec);
  auto endConst = rewriter.create<mlir::TF::ConstOp>(
      loc, endAttr.getType(), endAttr);

  auto stridesAttr = rewriter.getI32TensorAttr(stridesVec);
  auto stridesConst = rewriter.create<mlir::TFL::ConstOp>(
      loc, stridesAttr.getType(), stridesAttr);

  auto slicedX = rewriter.create<mlir::TFL::StridedSliceOp>(
      loc, slicedType, x, beginConst, endConst, stridesConst,
      beginMask, endMask, ellipsisMask, newAxisMask, shrinkAxisMask);

  return slicedX;
}

mlir::Value Concat(
    mlir::PatternRewriter &rewriter, const mlir::Location &loc,
    const std::vector<mlir::Value> &xs, uint32_t axis) {
  auto xShapes = std::vector< std::vector<int64_t> >();
  for (auto x : xs)
    xShapes.push_back(x.getType().dyn_cast<mlir::ShapedType>().getShape());

  auto numDims = xShapes.front().size();
  auto concatShapeVec = xShapes.front();
  for (int i = 0; i < numDims; ++i) {
    if (i == axis) {
      concatShapeVec[i] = 0;
      for (const auto &xShape : xShapes)
        concatShapeVec[i] += xShape[i];
    } else {
      auto dim = xShapes.front()[i];
      for (int j = 1; j < xShapes.size(); ++j)
        assert(xShapes[j][i] == dim);
    }
  }
  auto xType = xs.front().getType().dyn_cast<mlir::ShapedType>();
  auto concatType = xType.clone(concatShapeVec);

  auto concatX = rewriter.create<mlir::TFL::ConcatenationOp>(
      loc, concatType, xs, axis, "NONE");
  return concatX;
}

mlir::Value ScalarConst(
    mlir::PatternRewriter &rewriter, const mlir::Location &loc,
    float constant) {
  auto constShape = std::vector<int64_t>();
  auto constElementType = rewriter.getF32Type();
  auto constType = mlir::RankedTensorType::get(
      constShape, constElementType);
  auto constAttr = mlir::SplatElementsAttr::get(constType, constant);
  return rewriter.create<mlir::TFL::ConstOp>(
      loc, constAttr.getType(), constAttr);
}

mlir::Value PiecewiseRegression(
    mlir::PatternRewriter &rewriter, const mlir::Location &loc,
    const mlir::Value &x, const CalibrationData::Coeffs &coeffs) {
  const auto &[intercept, alphaHat, betaHats, breakpoints] = coeffs;
  auto xType = x.getType();
  auto alphaX = rewriter.create<mlir::TFL::MulOp>(
      loc, xType, x, ScalarConst(rewriter, loc, alphaHat),
      "NONE");
  auto shiftedX = rewriter.create<mlir::TFL::AddOp>(
      loc, xType, alphaX, ScalarConst(rewriter, loc, intercept), "NONE");
  auto result = shiftedX.getResult();
  auto n = betaHats.size();
  for (auto i = 0; i < n; ++i) {
    auto subX = rewriter.create<mlir::TFL::SubOp>(
        loc, xType, x,
        ScalarConst(rewriter, loc, breakpoints[i]), "RELU");
    auto betaX = rewriter.create<mlir::TFL::MulOp>(
        loc, xType, subX,
        ScalarConst(rewriter, loc, betaHats[i]), "NONE");
    result = rewriter.create<mlir::TFL::AddOp>(
        loc, xType, result, betaX, "NONE");
  }
  return result;
}

void AddMyTFPass(llvm::StringRef modelDir, mlir::OpPassManager *passManager) {
#ifdef MY_TF_PASS
  passManager->addPass(std::make_unique<MyPass::AllTFPasses>(
      modelDir.str()));
#endif
}

void AddMyTFLPass(llvm::StringRef modelDir, mlir::OpPassManager *passManager) {
#ifdef MY_TFL_PASS
  passManager->addPass(
      std::make_unique<MyPass::AllTFLPasses>(modelDir.str()));
#endif
}
} // namespace MyPass

namespace mlir {
/// Create a pass to convert from the TFExecutor to the TF control dialect.
std::unique_ptr<OperationPass<FuncOp>>
CreateTFExecutorToControlDialectConversion();
}  // namespace mlir

namespace tensorflow {
namespace {
// Data layout supported by TFLite.
const char kTFLiteDataLayout[] = "NHWC";
}  // namespace

void AddQuantizationPasses(const mlir::TFL::QuantizationSpecs& quant_specs,
                           mlir::OpPassManager* pass_manager) {
  pass_manager->addNestedPass<mlir::FuncOp>(
      mlir::TFL::CreatePrepareQuantizePass(quant_specs));
  if (quant_specs.default_ranges.first.hasValue() ||
      quant_specs.default_ranges.second.hasValue()) {
    pass_manager->addNestedPass<mlir::FuncOp>(
        mlir::TFL::CreateDefaultQuantParamsPass(
            quant_specs.default_ranges.first.getValueOr(0.0),
            quant_specs.default_ranges.second.getValueOr(0.0),
            quant_specs.IsSignedInferenceType()));
  }
  pass_manager->addNestedPass<mlir::FuncOp>(
      mlir::TFL::CreateQuantizePass(quant_specs));
  bool emit_quant_adaptor_ops =
      quant_specs.inference_type != quant_specs.inference_input_type;
  pass_manager->addNestedPass<mlir::FuncOp>(
      mlir::TFL::CreatePostQuantizePass(emit_quant_adaptor_ops));
}

void AddDynamicRangeQuantizationPasses(
    const mlir::TFL::QuantizationSpecs& quant_specs,
    mlir::OpPassManager& pass_manager) {
  pass_manager.addNestedPass<mlir::FuncOp>(
      mlir::TFL::CreatePrepareDynamicRangeQuantizePass(quant_specs));
  pass_manager.addNestedPass<mlir::FuncOp>(
      mlir::TFL::CreateQuantizePass(quant_specs));
  bool emit_quant_adaptor_ops =
      quant_specs.inference_type != quant_specs.inference_input_type;
  pass_manager.addNestedPass<mlir::FuncOp>(
      mlir::TFL::CreatePostQuantizePass(emit_quant_adaptor_ops));
}

void AddConvertHloToTfPass(std::string entry_function_name,
                           mlir::OpPassManager* pass_manager) {
  // Legalize jax random to tflite custom op.
  // The CreateLegalizeJaxRandom Pass has to stay at because we need to replace
  // the random function body before being inlined.
  pass_manager->addNestedPass<mlir::FuncOp>(
      mlir::TFL::CreateLegalizeJaxRandomPass());

  // Canonicalize, CSE etc.
  pass_manager->addNestedPass<mlir::FuncOp>(mlir::createCanonicalizerPass());
  pass_manager->addNestedPass<mlir::FuncOp>(mlir::createCSEPass());
  // DCE for private symbols.
  pass_manager->addPass(mlir::createSymbolDCEPass());

  pass_manager->addPass(mlir::TF::CreateStripNoinlineAttributePass());
  // Add inline pass.
  pass_manager->addPass(mlir::createInlinerPass());

  // Expands mhlo.tuple ops.
  pass_manager->addPass(
      mlir::mhlo::CreateExpandHloTuplesPass(entry_function_name));
  // Flatten tuples for control flows.
  pass_manager->addNestedPass<mlir::FuncOp>(
      mlir::mhlo::createFlattenTuplePass());

  // TF dialect passes
  pass_manager->addNestedPass<mlir::FuncOp>(
      mlir::TF::CreateLegalizeHloToTfPass());

  // Canonicalization after TF legalization.
  pass_manager->addNestedPass<mlir::FuncOp>(mlir::createCanonicalizerPass());
}

// This is the early part of the conversion in isolation. This enables a caller
// to inject more information in the middle of the conversion before resuming
// it.
void AddPreVariableFreezingTFToTFLConversionPasses(
    llvm::StringRef saved_model_dir, const toco::TocoFlags& toco_flags,
    const mlir::TFL::PassConfig& pass_config,
    mlir::OpPassManager* pass_manager) {
  if (pass_config.enable_hlo_to_tf_conversion) {
    // TODO(b/194747383): We need to valid that indeed the "main" func is
    // presented.
    AddConvertHloToTfPass("main", pass_manager);
  }
  // This pass wraps all the tf.FakeQuant ops in a custom op so they are not
  // folded before being converted to tfl.quantize and tfl.dequantize ops.
  auto wrapped_ops = mlir::TFL::AllTfFakeQuantOps();
  pass_manager->addNestedPass<mlir::FuncOp>(
      mlir::TFL::CreateRaiseCustomOpsPass(wrapped_ops));

  mlir::TF::StandardPipelineOptions standard_pipeline_options;
  standard_pipeline_options.enable_inliner = false;
  standard_pipeline_options.form_clusters = pass_config.form_clusters;
  mlir::TF::CreateTFStandardPipeline(*pass_manager, standard_pipeline_options);
  pass_manager->addNestedPass<mlir::FuncOp>(
      mlir::TF::CreateDeviceIndexSelectorPass());

  // Add canonicalize pass to remove no-op session initializer pass.
  pass_manager->addPass(mlir::createCanonicalizerPass());

  if (pass_config.guarantee_all_funcs_one_use) {
    pass_manager->addPass(mlir::TF::CreateGuaranteeAllFuncsOneUsePass());
  }
  if (pass_config.shape_inference) {
    pass_manager->addPass(mlir::TF::CreateTFShapeInferencePass());
  }

  // Keep this pass after the shape inference pass, which couldn't do shape
  // inference for non-tf ops.
  if (!pass_config.quant_specs.serialized_quant_stats.empty()) {
    pass_manager->addNestedPass<mlir::FuncOp>(
        mlir::quant::CreateImportQuantStatsPassForTFControlDialect(
            pass_config.quant_specs.serialized_quant_stats));
  }

  pass_manager->addPass(mlir::TF::CreateTFFunctionalControlFlowToRegions());

  // The conversion pipeline has to follow the following orders:
  // 1) Saved model related optimization like decompose resource ops
  // 2) Convert composite functions like lstm/rnns, along with proper function
  // inlining & dce.
  // 3) Lower static tensor list pass.

  // This decomposes resource ops like ResourceGather into read-variable op
  // followed by gather. This is used when the saved model import path is used
  // during which resources dont get frozen in the python layer.
  pass_manager->addNestedPass<mlir::FuncOp>(
      mlir::TFDevice::CreateDecomposeResourceOpsPass());
}

// This is the later part of the conversion in isolation. This enables a caller
// to resume the conversion after injecting more information in the middle of
// it.
void AddPostVariableFreezingTFToTFLConversionPasses(
    llvm::StringRef saved_model_dir, const toco::TocoFlags& toco_flags,
    const mlir::TFL::PassConfig& pass_config,
    mlir::OpPassManager* pass_manager) {
  // Note:
  // We need to fuse composite ops before LowerStaticTensorList pass.
  // The tensorflow list is not supported right now by that pass.
  // Enable fusing composite ops that can be lowered to built-in TFLite ops.
  if (pass_config.emit_builtin_tflite_ops &&
      toco_flags.tf_quantization_mode().empty()) {
    pass_manager->addPass(mlir::TFL::CreatePrepareCompositeFunctionsPass());
  }

  pass_manager->addPass(mlir::TF::CreateTFRegionControlFlowToFunctional());

  pass_manager->addPass(mlir::createInlinerPass());
  pass_manager->addPass(mlir::createSymbolDCEPass());

  MyPass::AddMyTFPass(saved_model_dir, pass_manager);

  if (pass_config.lower_tensor_list_ops &&
      toco_flags.tf_quantization_mode().empty()) {
    // TODO(haoliang): Add this pass by default.
    pass_manager->addPass(mlir::TFL::CreateLowerStaticTensorListPass(
        /*allow_tensorlist_pass_through=*/toco_flags.force_select_tf_ops() ||
            toco_flags.enable_select_tf_ops(),
        /*default_to_single_batch=*/toco_flags
            .default_to_single_batch_in_tensor_list_ops()));
  }

  // This pass does resource analysis of saved model global tensors and marks
  // those deemed read-only as immutable.
  pass_manager->addPass(
      mlir::tf_saved_model::CreateOptimizeGlobalTensorsPass());

  if (pass_config.shape_inference) {
    // Add a shape inference pass to optimize away the unnecessary casts.
    pass_manager->addPass(mlir::TF::CreateTFShapeInferencePass());
  }

  // Legalize while early to allow further constant folding.
  // TODO(jpienaar): This may not actually matter as we do canonicalization
  // after the legalize below, for now it needs to be below the above passes
  // that work on TF dialect and before inliner so that the function calls in
  // body and cond are inlined for optimization.
  if (pass_config.legalize_tf_while) {
    pass_manager->addPass(mlir::TFL::CreateLegalizeTFWhilePass());
  }

  // Add function inlining pass. Both TF and TFLite dialects are opted into
  // function inliner interface.
  pass_manager->addPass(mlir::createInlinerPass());
  // Reduce operands of TFL::While without changing the outcome.
  // It needs to stay here because:
  // 1. WhileOps are in TFL dialect.
  // 2. The body and cond are inlined.
  // 3. We need to do this before while canonicalization, otherwise it would be
  //   difficult to find dependencies.
  pass_manager->addNestedPass<mlir::FuncOp>(
      mlir::TFL::CreateReduceWhileOperandsPass());
  // Canonicalization includes const folding, which is utilized here to optimize
  // away ops that can't get constant folded after PrepareTF pass. For example,
  // tf.Conv2D is split into tf.Transpose and tfl.Conv2D.
  pass_manager->addNestedPass<mlir::FuncOp>(mlir::createCanonicalizerPass());
  pass_manager->addNestedPass<mlir::FuncOp>(mlir::createCSEPass());
  // This pass does dead code elimination based on symbol visibility.
  pass_manager->addPass(mlir::createSymbolDCEPass());

  if (!pass_config.disable_variable_freezing) {
    // This pass 'freezes' immutable global tensors and inlines them as tf
    // constant ops.
    pass_manager->addPass(mlir::tf_saved_model::CreateFreezeGlobalTensorsPass(
        /*allow_mutable_tensors=*/pass_config.enable_tflite_variables));
  }

  if (!saved_model_dir.empty()) {
    // This pass 'freezes' tf saved model asset ops and inlines as string values
    // in a format of the tf constant op.
    pass_manager->addPass(
        mlir::tf_saved_model::CreateFreezeAssetsPass(saved_model_dir.str()));
  }
  // For TF Quantization, convert unsupported ops to Flex ops before other
  // conversion passes.
  if (!toco_flags.tf_quantization_mode().empty()) {
    pass_manager->addNestedPass<mlir::FuncOp>(
        mlir::TF::CreateFallbackToFlexOpsPass(
            toco_flags.tf_quantization_mode()));
  }
  // The below passes only make sense if Builtin TFLite ops are enabled
  // for emission.
  if (pass_config.emit_builtin_tflite_ops) {
    // Run shape inference after variables are converted to constants.
    if (pass_config.shape_inference) {
      pass_manager->addPass(mlir::TF::CreateTFShapeInferencePass());
    }
    // Force layout supported by TFLite, this will transpose the data
    // to match 'kTFLiteDataLayout'
    mlir::TF::LayoutOptimizationPipelineOptions layout_optimization_options;
    layout_optimization_options.force_data_format = kTFLiteDataLayout;
    layout_optimization_options.skip_fold_transpose_in_ops = true;
    mlir::TF::CreateLayoutOptimizationPipeline(
        pass_manager->nest<mlir::FuncOp>(), layout_optimization_options);
    // Prepare for TFLite dialect, rerun canonicalization, and then legalize to
    // the TFLite dialect.
    pass_manager->addNestedPass<mlir::FuncOp>(
        mlir::TFL::CreatePrepareTFPass(pass_config.unfold_batch_matmul,
                                       /*allow_bf16_and_f16_type_legalization=*/
                                       !pass_config.runtime_verification,
                                       toco_flags.use_fake_quant_num_bits()));
    pass_manager->addNestedPass<mlir::FuncOp>(mlir::createCanonicalizerPass());
    if (pass_config.shape_inference) {
      // Add a shape inference pass to optimize away the unnecessary casts.
      // This also fixes the unranked shapes due to TF ops constant folding.
      // TODO(fengliuai): remove this pass if TableGen patterns have a better
      // to control the shapes for the intermediate results.
      pass_manager->addPass(mlir::TF::CreateTFShapeInferencePass());
    }

    // Inline function calls that left in the graph after folding functional
    // control flow ops (IfOp, CaseOp).
    pass_manager->addPass(mlir::createInlinerPass());

    // This pass removes the asset file dependencies in hash table use cases.
    pass_manager->addNestedPass<mlir::FuncOp>(
        mlir::TF::CreateInitTextFileToImportPass(saved_model_dir.str()));

    pass_manager->addNestedPass<mlir::FuncOp>(
        mlir::TFL::CreateLegalizeTFPass(pass_config.runtime_verification));
    pass_manager->addPass(mlir::TFL::CreateAnalyzeVariablesPass());
    pass_manager->addPass(mlir::TFL::CreateLegalizeVariablesPass());
    pass_manager->addPass(mlir::TFL::CreateLegalizeHashTablesPass());
    pass_manager->addNestedPass<mlir::FuncOp>(
        mlir::TFL::CreateOptimizePass(/*enable_canonicalization=*/true));
    // This pass operates on TensorFlow ops but is triggered after legalization
    // so that it can target constants introduced once TensorFlow Identity ops
    // are removed during legalization.
    pass_manager->addPass(mlir::TFL::CreateOptimizeFunctionalOpsPass());
    std::vector<std::string> empty_wrapped_ops({});
    pass_manager->addNestedPass<mlir::FuncOp>(
        mlir::TFL::CreateRaiseCustomOpsPass(empty_wrapped_ops));
    pass_manager->addPass(mlir::createSymbolDCEPass());
    pass_manager->addNestedPass<mlir::FuncOp>(mlir::createCanonicalizerPass());
    pass_manager->addNestedPass<mlir::FuncOp>(mlir::createCSEPass());

    MyPass::AddMyTFLPass(saved_model_dir, pass_manager);

    // Run quantization after all the floating point model conversion is
    // completed. Add either full integer quantization or dynamic range
    // quantization passes based on quant_specs.
    if (pass_config.quant_specs.RunPropagationAndRewriteQuantizationPasses()) {
      AddQuantizationPasses(pass_config.quant_specs, pass_manager);
    } else if (pass_config.quant_specs
                   .RunAndRewriteDynamicRangeQuantizationPasses()) {
      AddDynamicRangeQuantizationPasses(pass_config.quant_specs, *pass_manager);
    }
    pass_manager->addPass(mlir::createCanonicalizerPass());

    // This pass should be always at the end of the model
    // conversion (even after quantization). Some TFL ops like unidirectional
    // sequence lstm will have stateful operands and some optimization passes
    // will merge those operands if they have identical values & types. However,
    // it's not desired by TFL. This pass serves as a "fix" pass to split the
    // merged inputs until we have 1st class variable support or reuse
    // tf.variable to model this.
    pass_manager->addNestedPass<mlir::FuncOp>(
        mlir::TFL::CreateSplitMergedOperandsPass());

    // Add CallOnceOp when there is a session initializer function in tf saved
    // model dialect.
    pass_manager->addPass(
        mlir::TFL::CreateInsertCallOnceOpFromSessionInitializerPass());
  }
  if (pass_config.unfold_large_splat_constant) {
    pass_manager->addPass(mlir::TFL::CreateUnfoldLargeSplatConstantPass());
  }
  if (pass_config.outline_tf_while) {
    pass_manager->addPass(mlir::TFL::CreateWhileOutlinePass());
  }
  if (pass_config.runtime_verification) {
    pass_manager->addNestedPass<mlir::FuncOp>(
        mlir::TFL::CreateRuntimeVerifyPass());
  }
}

void AddTFToTFLConversionPasses(llvm::StringRef saved_model_dir,
                                const toco::TocoFlags& toco_flags,
                                const mlir::TFL::PassConfig& pass_config,
                                mlir::OpPassManager* pass_manager) {
  AddPreVariableFreezingTFToTFLConversionPasses(saved_model_dir, toco_flags,
                                                pass_config, pass_manager);
  AddPostVariableFreezingTFToTFLConversionPasses(saved_model_dir, toco_flags,
                                                 pass_config, pass_manager);
}
void AddTFToTFLConversionPasses(const mlir::TFL::PassConfig& pass_config,
                                mlir::OpPassManager* pass_manager) {
  const toco::TocoFlags toco_flags;
  AddTFToTFLConversionPasses(/*saved_model_dir=*/"", toco_flags, pass_config,
                             pass_manager);
}

}  // namespace tensorflow

namespace mlir {
namespace TFL {

struct StandardPipelineOptions
    : public PassPipelineOptions<StandardPipelineOptions> {
  // TODO(b/150915052): All the tf_tfl_translate_cl flags should
  // move inside this.
};

// NOLINTNEXTLINE
// This creates the standard pass pipeline for TF->TFLite. This
// represents a std configuration for TFLite, for use with APIs like
// tensorflow/python/pywrap_mlir.py::experimental_run_pass_pipeline
// This does not yet include quantization passes.
void CreateTFLStandardPipeline(OpPassManager& pm,
                               const StandardPipelineOptions& options) {
  OpPassManager& func_pm = pm.nest<FuncOp>();

  // tf_executor dialect passes - Cleaning up the IR.
  mlir::TF::StandardPipelineOptions standard_pipeline_options;
  mlir::TF::CreateTFStandardPipeline(func_pm, standard_pipeline_options);

  // This is needed for control flow support with TF TensorList.
  pm.addPass(mlir::TFL::CreateLowerStaticTensorListPass(
      /*allow_tensorlist_pass_through=*/false,
      /*default_to_single_batch=*/false));

  // Saved model pass to mark global tensors immutable.
  pm.addPass(mlir::tf_saved_model::CreateOptimizeGlobalTensorsPass());
  // Op fusion pass.
  pm.addPass(mlir::TFL::CreatePrepareCompositeFunctionsPass());

  pm.addNestedPass<mlir::FuncOp>(mlir::TFL::CreateLegalizeTFWhilePass());

  pm.addPass(mlir::createInlinerPass());

  // Canonicalize, CSE etc.
  pm.addNestedPass<mlir::FuncOp>(mlir::createCanonicalizerPass());
  pm.addNestedPass<mlir::FuncOp>(mlir::createCSEPass());
  // DCE for private symbols.
  pm.addPass(mlir::createSymbolDCEPass());

  // freeze global tensors.
  pm.addPass(mlir::tf_saved_model::CreateFreezeGlobalTensorsPass());

  // TFLite dialect passes.
  pm.addPass(mlir::TFL::CreatePrepareTFPass(
      /*unfold_batch_matmul=*/true,
      /*allow_bf16_and_f16_type_legalization=*/false));
  pm.addNestedPass<mlir::FuncOp>(mlir::createCanonicalizerPass());
  pm.addPass(
      mlir::TFL::CreateLegalizeTFPass(/*run_tfl_runtime_verification=*/true));
  pm.addPass(mlir::TFL::CreateLegalizeHashTablesPass());
  pm.addPass(mlir::TFL::CreateOptimizePass(/*enable_canonicalization=*/true));
  pm.addPass(mlir::TFL::CreateOptimizeFunctionalOpsPass());
  pm.addPass(mlir::createSymbolDCEPass());

  // Canonicalize, CSE etc.
  pm.addNestedPass<mlir::FuncOp>(mlir::createCanonicalizerPass());
  pm.addNestedPass<mlir::tf_saved_model::SessionInitializerOp>(
      mlir::createCanonicalizerPass());
  pm.addNestedPass<mlir::FuncOp>(mlir::createCSEPass());

  // Pass for stateful operands like LSTM.
  pm.addPass(mlir::TFL::CreateSplitMergedOperandsPass());

  pm.addPass(mlir::TFL::CreateWhileOutlinePass());

  pm.addNestedPass<mlir::FuncOp>(mlir::TFL::CreateRuntimeVerifyPass());
}

// Registers a pass pipeline for the standard TFL passes.
static mlir::PassPipelineRegistration<StandardPipelineOptions> pipeline(
    "tfl-standard-pipeline",
    "Run the standard passes involved in transforming/optimizing the TF "
    "program to TFLite after "
    "importing into MLIR.",
    CreateTFLStandardPipeline);

}  // namespace TFL
}  // namespace mlir
