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

#include "llvm/ADT/Optional.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/quantization/quantization_config.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_passes.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/decode_constant.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_saved_model_passes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include <fstream>
#include <cfloat>
#include <cstdlib>

namespace mlir {
/// Create a pass to convert from the TFExecutor to the TF control dialect.
std::unique_ptr<OperationPass<FuncOp>>
CreateTFExecutorToControlDialectConversion();
}  // namespace mlir

namespace MyPass {
struct ReplaceAbsOp : public mlir::OpRewritePattern<mlir::TFL::AbsOp> {
  explicit ReplaceAbsOp(mlir::MLIRContext *ctx)
    : mlir::OpRewritePattern<mlir::TFL::AbsOp>(ctx, /*benefit=*/1) {}

  mlir::LogicalResult matchAndRewrite(
      mlir::TFL::AbsOp op, mlir::PatternRewriter &rewriter) const override {
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
};

#define LEAKY_RELU_X_ALPHAX
struct ReplaceLeakyReluOp : public mlir::OpRewritePattern<
    mlir::TFL::LeakyReluOp> {
  explicit ReplaceLeakyReluOp(mlir::MLIRContext *ctx)
    : mlir::OpRewritePattern<mlir::TFL::LeakyReluOp>(ctx, /*benefit=*/1) {}

  mlir::LogicalResult matchAndRewrite(
      mlir::TFL::LeakyReluOp op,
      mlir::PatternRewriter &rewriter) const override {
    llvm::dbgs() << "INFO: LeakyReluOp is called!\n";

    auto x = op.getOperand();

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
    rewriter.replaceOpWithNewOp<mlir::TFL::PReluOp>(op, xType, x, alphaConst);
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
};

struct ReplaceTileOp : public mlir::OpRewritePattern<mlir::TFL::TileOp> {
  explicit ReplaceTileOp(mlir::MLIRContext *ctx)
      : mlir::OpRewritePattern<mlir::TFL::TileOp>(ctx, /*benefit=*/1) {}

  mlir::LogicalResult matchAndRewrite(
      mlir::TFL::TileOp op, mlir::PatternRewriter &rewriter) const override {
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
    for (auto it = multiplesAttr.int_value_begin(),
         end = multiplesAttr.int_value_end(); it != end; ++it) {
      auto multiple = (*it).getLimitedValue();
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
};

#define UNPACK_TFL
#ifdef UNPACK_TFL
struct ReplaceUnpackOp : public mlir::OpRewritePattern<mlir::TFL::UnpackOp> {
  explicit ReplaceUnpackOp(mlir::MLIRContext *ctx)
    : mlir::OpRewritePattern<mlir::TFL::UnpackOp>(ctx, /*benefit=*/1) {}

  mlir::LogicalResult matchAndRewrite(
      mlir::TFL::UnpackOp op, mlir::PatternRewriter &rewriter) const override {
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
      auto endVals = std::vector<int32_t>(xType.getShape().size(), 0);
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
};
#endif

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

#define EXP_LEAST_SQ

struct ReplaceExpOp : public mlir::OpRewritePattern<mlir::TFL::ExpOp> {
  explicit ReplaceExpOp(mlir::MLIRContext *ctx)
    : mlir::OpRewritePattern<mlir::TFL::ExpOp>(ctx, /*benefit=*/1) {}

  mlir::LogicalResult matchAndRewrite(
      mlir::TFL::ExpOp op, mlir::PatternRewriter &rewriter) const override {
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
};

#define LOG_LEAST_SQ

struct ReplaceLogOp : public mlir::OpRewritePattern<mlir::TFL::LogOp> {
  explicit ReplaceLogOp(mlir::MLIRContext *ctx)
    : mlir::OpRewritePattern<mlir::TFL::LogOp>(ctx, /*benefit=*/1) {}

  mlir::LogicalResult matchAndRewrite(
      mlir::TFL::LogOp op, mlir::PatternRewriter &rewriter) const override {
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
};

#define MY_TFL_PASS
struct AllTFLPasses : public mlir::PassWrapper<AllTFLPasses,
                                               mlir::OperationPass<>> {
  void runOnOperation() override {
    auto ctx = &getContext();
    mlir::RewritePatternSet patterns(ctx);
    patterns.insert(std::make_unique<ReplaceAbsOp>(ctx));
    patterns.insert(std::make_unique<ReplaceLeakyReluOp>(ctx));
    patterns.insert(std::make_unique<ReplaceTileOp>(ctx));
#ifdef UNPACK_TFL
    patterns.insert(std::make_unique<ReplaceUnpackOp>(ctx));
#endif
    patterns.insert(std::make_unique<ReplaceExpOp>(ctx));
    patterns.insert(std::make_unique<ReplaceLogOp>(ctx));

    mlir::applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

class CalibrationData {
private:
  constexpr static const char *coeffCalculatorPath_ =
      "/home/robert/compiler-lab/edgetpu-pass/src/tools/"
      "poly_approx_get_coeff.py";
  std::string path_;
  std::unordered_map<std::string, std::pair<float, float>> data_;

public:
  explicit CalibrationData(const std::string &path, const std::string &filename)
    : path_(path) {
    std::fstream file(path + "/" + filename);
    std::string opName;
    while (file >> opName) {
      float minVal, maxVal;
      file >> minVal >> maxVal;
      data_[opName] = std::make_pair(minVal, maxVal);
    }
  }

  std::pair<float, float> getRange(mlir::Location loc) const {
    mlir::NameLoc nameLoc;
    while (!(nameLoc = loc.dyn_cast<mlir::NameLoc>())) {
      mlir::CallSiteLoc callSiteLoc;
      if (callSiteLoc = loc.dyn_cast<mlir::CallSiteLoc>())
        loc = callSiteLoc.getCallee();
      else
        return std::make_pair(FLT_MIN, FLT_MAX);
    }
    std::string locName(nameLoc.getName().c_str());
    std::string opName = locName.substr(
        0, locName.find_first_of('@')) + ":0";

    auto it = data_.find(opName);
    if (it == data_.end())
      return std::make_pair(FLT_MIN, FLT_MAX);
    return it->second;
  }

  void getPolyCoeff(mlir::Location loc,
                    const std::string &function, const std::string &method,
                    int n, float coeffs[]) const {
    auto [minVal, maxVal] = getRange(loc);
    std::string filePath = path_ + "/" + function + "_poly.txt";
    std::string cmd = std::string(coeffCalculatorPath_) +
                      " " + function +
                      " " + method +
                      " " + std::to_string(n) +
                      " " + std::to_string(minVal) +
                      " " + std::to_string(maxVal) +
                      " " + filePath;
    std::system(cmd.c_str());

    std::fstream file(filePath);
    for (int i = 0; i <= n; ++i)
      file >> coeffs[i];

    llvm::dbgs() << "INFO: range: (" << minVal << ", " << maxVal << "), "
                 << "calibrated polynomial:";
    for (int i = 0; i <= n; ++i)
      llvm::dbgs() << " " << coeffs[i];
    llvm::dbgs() << "\n";
  }
};

#define SOFTPLUS_RELU
struct ReplaceSoftplusOp : public mlir::OpRewritePattern<mlir::TF::SoftplusOp> {
  CalibrationData calibrationData_;

  ReplaceSoftplusOp(mlir::MLIRContext *ctx, const std::string &savedModelPath)
    : mlir::OpRewritePattern<mlir::TF::SoftplusOp>(ctx, /*benefit=*/1),
      calibrationData_(savedModelPath, "Softplus_calibration.txt") {}

  mlir::LogicalResult matchAndRewrite(
      mlir::TF::SoftplusOp op, mlir::PatternRewriter &rewriter) const override {
    llvm::dbgs() << "INFO: SoftplusOp is called!\n";

#ifdef SOFTPLUS_X
    rewriter.replaceOp(op, {op.getOperand()});
#endif

#ifdef SOFTPLUS_RELU
    rewriter.replaceOpWithNewOp<mlir::TF::ReluOp>(op, op.getOperand());
#endif

#ifdef SOFTPLUS_LINE_SEGMENTS
    const float a = -1.7, b = 1.5, m = b / (b - a), n = 1 - m;

    auto x = op.getOperand();
    auto xType = x.getType().dyn_cast<mlir::ShapedType>();
    // TODO: check out broadcasting
    auto aAttr = mlir::SplatElementsAttr::get(xType, a);
    auto aConst = rewriter.create<mlir::TF::ConstOp>(
        op.getLoc(), aAttr.getType(), aAttr);
    auto xSuba = rewriter.create<mlir::TF::SubOp>(op.getLoc(), x, aConst);
    auto relu1 = rewriter.create<mlir::TF::ReluOp>(op.getLoc(), xSuba);
    auto mAttr = mlir::SplatElementsAttr::get(xType, m);
    auto mConst = rewriter.create<mlir::TF::ConstOp>(
        op.getLoc(), mAttr.getType(), mAttr);
    auto part1 = rewriter.create<mlir::TF::MulOp>(op.getLoc(), mConst, relu1);

    auto bAttr = mlir::SplatElementsAttr::get(xType, b);
    auto bConst = rewriter.create<mlir::TF::ConstOp>(
        op.getLoc(), bAttr.getType(), bAttr);
    auto xSubb = rewriter.create<mlir::TF::SubOp>(op.getLoc(), x, bConst);
    auto relu2 = rewriter.create<mlir::TF::ReluOp>(op.getLoc(), xSubb);
    auto nAttr = mlir::SplatElementsAttr::get(xType, n);
    auto nConst = rewriter.create<mlir::TF::ConstOp>(
        op.getLoc(), nAttr.getType(), nAttr);
    auto part2 = rewriter.create<mlir::TF::MulOp>(op.getLoc(), nConst, relu2);

    rewriter.replaceOpWithNewOp<mlir::TF::AddOp>(op, part1, part2);
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
};

#ifndef UNPACK_TFL
struct ReplaceUnpackOp : public mlir::OpRewritePattern<mlir::TF::UnpackOp> {
  explicit ReplaceUnpackOp(mlir::MLIRContext *ctx)
      : mlir::OpRewritePattern<mlir::TF::UnpackOp>(ctx, /*benefit=*/1) {}

  mlir::LogicalResult matchAndRewrite(
      mlir::TF::UnpackOp op, mlir::PatternRewriter &rewriter) const override {
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
      auto endVals = std::vector<int32_t>(xType.getShape().size(), 0);
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
};
#endif

#define MY_TF_PASS
struct AllTFPasses : public mlir::PassWrapper<AllTFPasses,
                                              mlir::OperationPass<>> {
public:
  explicit AllTFPasses(const std::string &savedModelPath)
    : savedModelPath_(savedModelPath) {}

private:
  std::string savedModelPath_;

  void runOnOperation() override {
    auto ctx = &getContext();
    mlir::RewritePatternSet patterns(ctx);
    patterns.insert(std::make_unique<ReplaceSoftplusOp>(
        ctx, savedModelPath_));
#ifndef UNPACK_TFL
    patterns.insert(std::make_unique<ReplaceUnpackOp>(ctx));
#endif

    mlir::applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};
}

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
      mlir::TFL::CreateQuantizePass(quant_specs.verify_numeric));
  bool emit_quant_adaptor_ops =
      quant_specs.inference_type != quant_specs.inference_input_type;
  pass_manager->addNestedPass<mlir::FuncOp>(
      mlir::TFL::CreatePostQuantizePass(emit_quant_adaptor_ops));
}

void AddTFToTFLConversionPasses(const toco::ModelFlags& model_flags,
                                const toco::TocoFlags& toco_flags,
                                const mlir::TFL::PassConfig& pass_config,
                                mlir::OpPassManager* pass_manager,
                                llvm::Optional<tensorflow::Session*> session) {
  mlir::TF::StandardPipelineOptions standard_pipeline_options;
  standard_pipeline_options.enable_inliner = false;
  standard_pipeline_options.form_clusters = pass_config.form_clusters;
  mlir::TF::CreateTFStandardPipeline(*pass_manager, standard_pipeline_options);
  pass_manager->addNestedPass<mlir::FuncOp>(
      mlir::TF::CreateDeviceIndexSelectorPass());

  // Add canonicalize pass to remove no-op session initializer pass.
  pass_manager->addPass(mlir::createCanonicalizerPass());

  if (pass_config.shape_inference) {
    pass_manager->addPass(mlir::TF::CreateTFShapeInferencePass());
  }

  if (session.hasValue()) {
    // Add a pass that converts reference variables to resource variables.
    pass_manager->addPass(
        mlir::TF::
            CreateConvertReadonlyReferenceVariablesToResourceVariablesPass());

    // Add a pass that promotes resource variable to the function arguments.
    pass_manager->addPass(mlir::TF::CreatePromoteVarHandlesToArgsPass());

    // Add a pass that creates global tensors and converts the function
    // arguments to the tf_saved_model.bound_input arguments.
    pass_manager->addPass(
        mlir::tf_saved_model::CreateLiftVariablesPass(session.getValue()));
  }

  // Keep this pass after the shape inference pass, which couldn't do shape
  // inference for non-tf ops.
  if (!pass_config.quant_specs.serialized_quant_stats.empty()) {
    pass_manager->addPass(
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

  // Note:
  // We need to fuse composite ops before LowerStaticTensorList pass.
  // The tensorflow list is not supported right now by that pass.
  // Enable fusing composite ops that can be lowered to built-in TFLite ops.
  if (pass_config.emit_builtin_tflite_ops) {
    pass_manager->addPass(mlir::TFL::CreatePrepareCompositeFunctionsPass());
  }

  pass_manager->addPass(mlir::TF::CreateTFRegionControlFlowToFunctional());

  pass_manager->addPass(mlir::createInlinerPass());
  pass_manager->addPass(mlir::createSymbolDCEPass());

  if (pass_config.lower_tensor_list_ops) {
    // TODO(haoliang): Add this pass by default.
    pass_manager->addPass(mlir::TFL::CreateLowerStaticTensorListPass(
        /*allow_tensorlist_pass_through=*/toco_flags.force_select_tf_ops() ||
        toco_flags.enable_select_tf_ops()));
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

  // TODO(jpienaar): Revise post dialect constants.
  pass_manager->addNestedPass<mlir::FuncOp>(
      mlir::TF::CreateDecodeConstantPass());
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

  if (!model_flags.saved_model_dir().empty()) {
    // This pass 'freezes' tf saved model asset ops and inlines as string values
    // in a format of the tf constant op.
    pass_manager->addPass(mlir::tf_saved_model::CreateFreezeAssetsPass(
        model_flags.saved_model_dir()));
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
    pass_manager->addNestedPass<mlir::FuncOp>(mlir::TFL::CreatePrepareTFPass(
        pass_config.unfold_batch_matmul,
        /*allow_bf16_and_f16_type_legalization=*/!pass_config
            .runtime_verification));
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
        mlir::TF::CreateInitTextFileToImportPass());

#ifdef MY_TF_PASS
    llvm::dbgs() << "INFO: my TF pass is enabled.\n";
    pass_manager->addPass(std::make_unique<MyPass::AllTFPasses>(
        model_flags.saved_model_dir()));
#endif

    pass_manager->addNestedPass<mlir::FuncOp>(
        mlir::TFL::CreateLegalizeTFPass(pass_config.runtime_verification));
    if (pass_config.enable_tflite_variables) {
      pass_manager->addPass(mlir::TFL::CreateInitializeVariablesPass());
      pass_manager->addPass(mlir::TFL::CreateLegalizeVariablesPass());
      pass_manager->addPass(mlir::TFL::CreateRemoveArgsAndGlobalTensors());
    }
    pass_manager->addPass(mlir::TFL::CreateLegalizeHashTablesPass());
    pass_manager->addNestedPass<mlir::FuncOp>(
        mlir::TFL::CreateOptimizePass(/*enable_canonicalization=*/true));
    // This pass operates on TensorFlow ops but is triggered after legalization
    // so that it can target constants introduced once TensorFlow Identity ops
    // are removed during legalization.
    pass_manager->addPass(mlir::TFL::CreateOptimizeFunctionalOpsPass());
    pass_manager->addNestedPass<mlir::FuncOp>(
        mlir::TFL::CreateRaiseCustomOpsPass());
    pass_manager->addPass(mlir::createSymbolDCEPass());
    pass_manager->addNestedPass<mlir::FuncOp>(mlir::createCanonicalizerPass());
    pass_manager->addNestedPass<mlir::FuncOp>(mlir::createCSEPass());

#ifdef MY_TFL_PASS
    llvm::dbgs() << "INFO: my TFL pass is enabled.\n";
    pass_manager->addPass(std::make_unique<MyPass::AllTFLPasses>());
#endif

    // Run quantization after all the floating point model conversion is
    // completed.
    if (pass_config.quant_specs.RunPropagationAndRewriteQuantizationPasses())
      AddQuantizationPasses(pass_config.quant_specs, pass_manager);

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
}

void AddTFToTFLConversionPasses(const mlir::TFL::PassConfig& pass_config,
                                mlir::OpPassManager* pass_manager,
                                llvm::Optional<tensorflow::Session*> session) {
  const toco::ModelFlags model_flags;
  const toco::TocoFlags toco_flags;
  AddTFToTFLConversionPasses(model_flags, toco_flags, pass_config, pass_manager,
                             session);
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
      /*allow_tensorlist_pass_through=*/false));

  // Saved model pass to mark global tensors immutable.
  pm.addPass(mlir::tf_saved_model::CreateOptimizeGlobalTensorsPass());
  // Op fusion pass.
  pm.addPass(mlir::TFL::CreatePrepareCompositeFunctionsPass());

  pm.addNestedPass<mlir::FuncOp>(mlir::TFL::CreateLegalizeTFWhilePass());

  pm.addPass(mlir::createInlinerPass());

  // Canonicalize, CSE etc.
  pm.addPass(mlir::TF::CreateDecodeConstantPass());
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
