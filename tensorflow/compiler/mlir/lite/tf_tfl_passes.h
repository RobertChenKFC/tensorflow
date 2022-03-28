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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_TF_TFL_PASSES_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_TF_TFL_PASSES_H_

#include "llvm/ADT/Optional.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/common/tfl_pass_config.h"
#include "tensorflow/lite/toco/model_flags.pb.h"
#include "tensorflow/lite/toco/toco_flags.pb.h"
// My includes
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include <fstream>
#include <cfloat>
#include <cstdlib>

namespace tensorflow {

// Add the TF to TFLite passes, specified in the pass_config, into a
// pass_manager. The session object will be provided when the TF MLIR is
// imported from saved model version one and utilized for capturing resource
// variables. If the `saved_model_dir` directory path is provided, then the
// `tf_saved_model.asset` ops will be freezed.
void AddTFToTFLConversionPasses(llvm::StringRef saved_model_dir,
                                const toco::TocoFlags& toco_flags,
                                const mlir::TFL::PassConfig& pass_config,
                                mlir::OpPassManager* pass_manager);

// This is the early part of the conversion in isolation. This enables a caller
// to inject more information in the middle of the conversion before resuming it
// (like freezing variables for example).
void AddPreVariableFreezingTFToTFLConversionPasses(
    llvm::StringRef saved_model_dir, const toco::TocoFlags& toco_flags,
    const mlir::TFL::PassConfig& pass_config,
    mlir::OpPassManager* pass_manager);

// This is the later part of the conversion in isolation. This enables a caller
// to resume the conversion after injecting more information in the middle of
// it.
void AddPostVariableFreezingTFToTFLConversionPasses(
    llvm::StringRef saved_model_dir, const toco::TocoFlags& toco_flags,
    const mlir::TFL::PassConfig& pass_config,
    mlir::OpPassManager* pass_manager);

// Simplified API for TF->TFLite conversion with default flags.
void AddTFToTFLConversionPasses(const mlir::TFL::PassConfig& pass_config,
                                mlir::OpPassManager* pass_manager);

// Add the Quantization passes, specified in the quant_specs, into a pass
// manager.
void AddQuantizationPasses(const mlir::TFL::QuantizationSpecs& quant_specs,
                           mlir::OpPassManager* pass_manager);

// Add the DynamicRangeQuantization passes, specified in the quant_specs, into a
// pass manager.
void AddDynamicRangeQuantizationPasses(
    const mlir::TFL::QuantizationSpecs& quant_specs,
    mlir::OpPassManager& pass_manager);
}  // namespace tensorflow

namespace MyPass {

// Some passes have multiple methods of replacing the operations, so we use
// preprocessor flags to select between them. Only one of them should be
// selected for each pass. This area is dedicated to place these flags
// together so that it is easier to see if only one is selected.
// ============================================================================
// Leaky ReLU
#define LEAKY_RELU_PRELU
//#define LEAKY_RELU_X_ALPHAX
//#define LEAKY_RELU_POS_NEG

// Unpack
#define UNPACK_TFL
//#define UNPACK_TF

// Exp
#define EXP_DISABLE
//#define EXP_LEAST_SQ
//#define EXP_REL_LEAST_SQ
//#define EXP_MINIMAX
//#define EXP_TAYLOR

// Log
#define LOG_DISABLE
//#define LOG_LEAST_SQ
//#define LOG_REL_LEAST_SQ
//#define LOG_MINIMAX
//#define LOG_TAYLOR

// Mean
#define MEAN_ONE_AXIS
//#define MEAN_SUM_DIV
//#define MEAN_SUM_DIV_REPEAT
//#define MEAN_RESHAPE

// Softplus
//#define SOFTPLUS_TANH_MUL_RELU
//#define SOFTPLUS_TANH_MUL_SIGMOID_MUL
//#define SOFTPLUS_TANH_MUL_SIGMOID_NO_MUL
#define SOFTPLUS_TANH_MUL_LINE_SEGMENTS
//#define SOFTPLUS_RELU
//#define SOFTPLUS_X
//#define SOFTPLUS_LINE_SEGMENTS
//#define SOFTPLUS_LEAST_SQ
//#define SOFTPLUS_REL_LEAST_SQ
//#define SOFTPLUS_MINIMAX
//#define SOFTPLUS_TAYLOR

// All TensorFlow Lite passes
#define MY_TFL_PASS

// All TensorFlow passes
#define MY_TF_PASS

// ============================================================================

class CalibrationData {
public:
  using Coeffs = std::tuple<
      float, float, std::vector<float>, std::vector<float>>;

  explicit CalibrationData(
      const std::string &path, const std::string &filename);

  Coeffs getCoeffs(mlir::Location loc) const;

private:
  std::string path_;
  std::unordered_map<std::string, Coeffs> data_;
};

struct ReplaceAbsOp : public mlir::OpRewritePattern<mlir::TFL::AbsOp> {
  explicit ReplaceAbsOp(mlir::MLIRContext *ctx);

  mlir::LogicalResult matchAndRewrite(
      mlir::TFL::AbsOp op, mlir::PatternRewriter &rewriter) const override;
};

struct ReplaceLeakyReluOp : public mlir::OpRewritePattern<
    mlir::TFL::LeakyReluOp> {
  explicit ReplaceLeakyReluOp(mlir::MLIRContext *ctx);

  mlir::LogicalResult matchAndRewrite(
      mlir::TFL::LeakyReluOp op,
      mlir::PatternRewriter &rewriter) const override;
};

struct ReplaceTileOp : public mlir::OpRewritePattern<mlir::TFL::TileOp> {
  explicit ReplaceTileOp(mlir::MLIRContext *ctx);

  mlir::LogicalResult matchAndRewrite(
      mlir::TFL::TileOp op, mlir::PatternRewriter &rewriter) const override;
};

struct ReplaceTFLUnpackOp : public mlir::OpRewritePattern<mlir::TFL::UnpackOp> {
  explicit ReplaceTFLUnpackOp(mlir::MLIRContext *ctx);

  mlir::LogicalResult matchAndRewrite(
      mlir::TFL::UnpackOp op, mlir::PatternRewriter &rewriter) const override;
};

#ifndef EXP_DISABLE
struct ReplaceExpOp : public mlir::OpRewritePattern<mlir::TFL::ExpOp> {
  explicit ReplaceExpOp(mlir::MLIRContext *ctx);

  mlir::LogicalResult matchAndRewrite(
      mlir::TFL::ExpOp op, mlir::PatternRewriter &rewriter) const override;
};
#endif

#ifndef LOG_DISABLE
struct ReplaceLogOp : public mlir::OpRewritePattern<mlir::TFL::LogOp> {
  explicit ReplaceLogOp(mlir::MLIRContext *ctx);

  mlir::LogicalResult matchAndRewrite(
      mlir::TFL::LogOp op, mlir::PatternRewriter &rewriter) const override;
};
#endif

struct ReplaceFullyConnectedOp : public mlir::OpRewritePattern<
    mlir::TFL::FullyConnectedOp> {
  explicit ReplaceFullyConnectedOp(mlir::MLIRContext *ctx);

  mlir::LogicalResult matchAndRewrite(
      mlir::TFL::FullyConnectedOp op,
      mlir::PatternRewriter &rewriter) const override;
};

struct ReplaceSplitOp : public mlir::OpRewritePattern<
    mlir::TFL::SplitOp> {
  explicit ReplaceSplitOp(mlir::MLIRContext *ctx);

  mlir::LogicalResult matchAndRewrite(
      mlir::TFL::SplitOp op,
      mlir::PatternRewriter &rewriter) const override;
};

struct ReplaceMeanOp : public mlir::OpRewritePattern<
    mlir::TFL::MeanOp> {
  explicit ReplaceMeanOp(mlir::MLIRContext *ctx);

  mlir::LogicalResult matchAndRewrite(
      mlir::TFL::MeanOp op,
      mlir::PatternRewriter &rewriter) const override;
};

struct ReplaceSquaredDifferenceOp : public mlir::OpRewritePattern<
    mlir::TFL::SquaredDifferenceOp> {
  explicit ReplaceSquaredDifferenceOp(mlir::MLIRContext *ctx);

  mlir::LogicalResult matchAndRewrite(
      mlir::TFL::SquaredDifferenceOp op,
      mlir::PatternRewriter &rewriter) const override;
};

struct ReplaceNegOp : public mlir::OpRewritePattern<
    mlir::TFL::NegOp> {
  explicit ReplaceNegOp(mlir::MLIRContext *ctx);

  mlir::LogicalResult matchAndRewrite(
      mlir::TFL::NegOp op,
      mlir::PatternRewriter &rewriter) const override;
};

struct ReplaceResizeNearestNeighbor : public mlir::OpRewritePattern<
    mlir::TFL::ResizeNearestNeighborOp> {
  explicit ReplaceResizeNearestNeighbor(mlir::MLIRContext *ctx);

  mlir::LogicalResult matchAndRewrite(
      mlir::TFL::ResizeNearestNeighborOp op,
      mlir::PatternRewriter &rewriter) const override;
};

struct ReplaceResizeBilinear : public mlir::OpRewritePattern<
    mlir::TFL::ResizeBilinearOp> {
  explicit ReplaceResizeBilinear(mlir::MLIRContext *ctx);

  mlir::LogicalResult matchAndRewrite(
      mlir::TFL::ResizeBilinearOp op,
      mlir::PatternRewriter &rewriter) const override;
};

struct ReplaceHardSwishOp : public mlir::OpRewritePattern<
    mlir::TFL::HardSwishOp> {
  explicit ReplaceHardSwishOp(mlir::MLIRContext *ctx);

  mlir::LogicalResult matchAndRewrite(
      mlir::TFL::HardSwishOp op,
      mlir::PatternRewriter &rewriter) const override;
};

struct ReplaceDivOp : public mlir::OpRewritePattern<
    mlir::TFL::DivOp> {
  explicit ReplaceDivOp(mlir::MLIRContext *ctx);

  mlir::LogicalResult matchAndRewrite(
      mlir::TFL::DivOp op,
      mlir::PatternRewriter &rewriter) const override;
};

struct ReplaceSumOp : public mlir::OpRewritePattern<
    mlir::TFL::SumOp> {
  explicit ReplaceSumOp(mlir::MLIRContext *ctx);

  mlir::LogicalResult matchAndRewrite(
      mlir::TFL::SumOp op,
      mlir::PatternRewriter &rewriter) const override;
};

struct ReplaceExpSum : public mlir::RewritePattern {
  ReplaceExpSum(mlir::MLIRContext *ctx, mlir::PatternBenefit benefit = 1);

  mlir::LogicalResult match(mlir::Operation *op) const override;

  void rewrite(
      mlir::Operation *op, mlir::PatternRewriter &rewriter) const override;
};

struct ReplaceMaxPool2DOp : public
                            mlir::OpRewritePattern<mlir::TFL::MaxPool2DOp> {
  explicit ReplaceMaxPool2DOp(mlir::MLIRContext *ctx);

  mlir::LogicalResult matchAndRewrite(
      mlir::TFL::MaxPool2DOp op,
      mlir::PatternRewriter &rewriter) const override;
};

struct ReplaceAveragePool2DOp :
    public mlir::OpRewritePattern<mlir::TFL::AveragePool2DOp> {
  explicit ReplaceAveragePool2DOp(mlir::MLIRContext *ctx);

  mlir::LogicalResult matchAndRewrite(
      mlir::TFL::AveragePool2DOp op,
      mlir::PatternRewriter &rewriter) const override;
};

struct ReplaceReshapeOp : public mlir::OpRewritePattern<mlir::TFL::ReshapeOp> {
  explicit ReplaceReshapeOp(mlir::MLIRContext *ctx);

  mlir::LogicalResult matchAndRewrite(
      mlir::TFL::ReshapeOp op,
      mlir::PatternRewriter &rewriter) const override;
};

struct ReplaceMulPow : public mlir::RewritePattern {
  ReplaceMulPow(mlir::MLIRContext *ctx, mlir::PatternBenefit = 1);

  mlir::LogicalResult match(mlir::Operation *op) const override;

  void rewrite(
      mlir::Operation *op, mlir::PatternRewriter &rewriter) const override;
};

struct ReplacePowOp : public mlir::OpRewritePattern<mlir::TFL::PowOp> {
  explicit ReplacePowOp(mlir::MLIRContext *ctx);

  mlir::LogicalResult matchAndRewrite(
      mlir::TFL::PowOp op, mlir::PatternRewriter &rewriter) const override;
};

struct ReplaceMirrorPadOp : public mlir::OpRewritePattern<
    mlir::TFL::MirrorPadOp> {
  explicit ReplaceMirrorPadOp(mlir::MLIRContext *ctx);

  mlir::LogicalResult matchAndRewrite(
      mlir::TFL::MirrorPadOp op,
      mlir::PatternRewriter &rewriter) const override;
};

struct ReplaceSquareOp : public mlir::OpRewritePattern<mlir::TFL::SquareOp> {
  explicit ReplaceSquareOp(mlir::MLIRContext *ctx);

  mlir::LogicalResult matchAndRewrite(
      mlir::TFL::SquareOp op, mlir::PatternRewriter &rewriter) const override;
};

struct ReplaceSoftplusTanhMul : public mlir::RewritePattern {
  explicit ReplaceSoftplusTanhMul(
      mlir::MLIRContext *ctx, const std::string &savedModelPath,
      mlir::PatternBenefit benefit = 1);

  mlir::LogicalResult match(mlir::Operation *op) const override;

  void rewrite(
      mlir::Operation *op, mlir::PatternRewriter &rewriter) const override;

private:
  CalibrationData calibrationData_;
};

struct AllTFLPasses : public mlir::PassWrapper<AllTFLPasses,
    mlir::OperationPass<>> {
public:
  explicit AllTFLPasses(const std::string &savedModelPath);

private:
  std::string savedModelPath_;

  void runOnOperation() override;
};

struct ReplaceSoftplusOp : public mlir::OpRewritePattern<mlir::TF::SoftplusOp> {
  ReplaceSoftplusOp(mlir::MLIRContext *ctx, const std::string &savedModelPath);

  mlir::LogicalResult matchAndRewrite(
      mlir::TF::SoftplusOp op, mlir::PatternRewriter &rewriter) const override;

private:
  CalibrationData calibrationData_;
};

struct ReplaceTFUnpackOp : public mlir::OpRewritePattern<mlir::TF::UnpackOp> {
  explicit ReplaceTFUnpackOp(mlir::MLIRContext *ctx);

  mlir::LogicalResult matchAndRewrite(
      mlir::TF::UnpackOp op, mlir::PatternRewriter &rewriter) const override;
};

struct AllTFPasses : public mlir::PassWrapper<AllTFPasses,
    mlir::OperationPass<>> {
public:
  explicit AllTFPasses(const std::string &savedModelPath);

private:
  std::string savedModelPath_;

  void runOnOperation() override;
};

mlir::Value PolynomialValueTF(
    mlir::PatternRewriter &rewriter, const mlir::Location &loc,
    const mlir::Value &x, float coeffs[], int degree);

mlir::Value PolynomialValueTFL(
    mlir::PatternRewriter &rewriter, const mlir::Location &loc,
    const mlir::Value &x, float coeffs[], int degree);

mlir::Value Reshape(
    mlir::PatternRewriter &rewriter, const mlir::Location &loc,
    const mlir::Value &x, const mlir::ShapedType &xType,
    const std::vector<int64_t> &newXShapeVec);

mlir::Value StridedSlice(
    mlir::PatternRewriter &rewriter, const mlir::Location &loc,
    const mlir::Value &x, mlir::ShapedType xType,
    const std::vector<int64_t> &newXShapeVec,
    const std::vector<int32_t> &beginVec,
    const std::vector<int32_t> &endVec,
    const std::vector<int32_t> &stridesVec,
    int64_t beginMask = 0,
    int64_t endMask = 0,
    int64_t ellipsisMask = 0,
    int64_t newAxisMask = 0,
    int64_t shrinkAxisMask = 0);

mlir::Value Concat(
    mlir::PatternRewriter &rewriter, const mlir::Location &loc,
    const std::vector<mlir::Value> &xs, uint32_t axis);

mlir::Value ScalarConst(
    mlir::PatternRewriter &rewriter, const mlir::Location &loc,
    float constant);

template <class ResizeOp>
mlir::LogicalResult ResizePass(mlir::PatternRewriter &rewriter, ResizeOp op,
                               const std::vector<int64_t> &inputDimThreshs,
                               const std::vector<int64_t> &outputDimThreshs,
                               const std::vector<int32_t> &numSplits) {
  auto x = op.getOperand(0);
  auto xType = x.getType().template dyn_cast<mlir::ShapedType>();
  auto xShapeVec = xType.getShape().vec();
  auto resultType = op.getResult().getType().template dyn_cast<
      mlir::ShapedType>();
  auto resultShapeVec = resultType.getShape().vec();

  if (xShapeVec.size() != 4 || resultShapeVec.size() != 4)
    return mlir::failure();
  for (auto i = 0; i < 4; ++i) {
    if (xShapeVec[i] < inputDimThreshs[i] ||
        resultShapeVec[i] < outputDimThreshs[i])
      return mlir::failure();
  }

  auto begins = std::vector<std::vector<int32_t>>{{}};
  auto ends = std::vector<std::vector<int32_t>>{{}};
  auto resizedXShapes = std::vector<std::vector<int64_t>>{{}};
  auto numDims = 4;
  auto stride = std::vector<int32_t>(numDims, 1);
  for (auto i = 0; i < numDims; ++i) {
    auto numSplit = numSplits[i];
    auto dimSize = xShapeVec[i];
    auto size = dimSize / numSplit;
    auto resizedDimSize = resultShapeVec[i];
    auto resizedSize = resizedDimSize / numSplit;
    auto curBegins = decltype(begins)();
    auto curEnds = decltype(ends)();
    auto curResizedXShapes = decltype(resizedXShapes)();
    for (auto j = 0; j < begins.size(); ++j) {
      for (auto k = 0; k < numSplit; ++k) {
        auto begin = begins[j];
        auto end = ends[j];
        auto resizedXShape = resizedXShapes[j];
        begin.push_back(k * size);
        curBegins.push_back(begin);
        if (k == numSplit - 1) {
          end.push_back(dimSize);
          resizedXShape.push_back(resizedSize + resizedDimSize % numSplit);
        } else {
          end.push_back((k + 1) * size);
          resizedXShape.push_back(resizedSize);
        }
        curEnds.push_back(end);
        curResizedXShapes.push_back(resizedXShape);

        // DEBUG
        llvm::dbgs() << "axis: " << i << ", numSplit: " << numSplit << ", "
                     << "begin: ( ";
        for (auto a : begin)
          llvm::dbgs() << a << " ";
        llvm::dbgs() << "), end: ( ";
        for (auto a : end)
          llvm::dbgs() << a << " ";
        llvm::dbgs() << "), resized shape: (";
        for (auto a : resizedXShape)
          llvm::dbgs() << a << " ";
        llvm::dbgs() << ")\n";
      }
    }
    begins = std::move(curBegins);
    ends = std::move(curEnds);
    resizedXShapes = std::move(curResizedXShapes);
  }
  // DEBUG
  for (auto i = 0; i < begins.size(); ++i) {
    llvm::dbgs() << "i: " << i << ", begin ( ";
    for (auto j : begins[i])
      llvm::dbgs() << j << " ";
    llvm::dbgs() << "), end ( ";
    for (auto j : ends[i])
      llvm::dbgs() << j << " ";
    llvm::dbgs() << "), resized shape: (";
    for (auto a : resizedXShapes[i])
      llvm::dbgs() << a << " ";
    llvm::dbgs() << ")\n";
  }

  auto resizedXs = std::vector<mlir::Value>();
  for (auto i = 0; i < begins.size(); ++i) {
    auto subXShapeVec = std::vector<int64_t>();
    for (auto j = 0; j < numDims; ++j)
      subXShapeVec.push_back(ends[i][j] - begins[i][j]);
    auto subX = StridedSlice(
        rewriter, op.getLoc(), x, xType, subXShapeVec,
        begins[i], ends[i], stride);

    // DEBUG
    llvm::dbgs() << "subX: ";
    subX.dump();

    auto resizedXShapeVec = resizedXShapes[i];
    auto resizedXType = resultType.clone(resizedXShapeVec);
    auto xSizeVec = std::vector<int32_t>{
        static_cast<int32_t>(resizedXShapeVec[1]),
        static_cast<int32_t>(resizedXShapeVec[2])
    };
    auto xSizeAttr = rewriter.getI32TensorAttr(xSizeVec);
    auto xSize = rewriter.create<mlir::TFL::ConstOp>(
        op.getLoc(), xSizeAttr.getType(), xSizeAttr);

    // DEBUG
    llvm::dbgs() << "xSize: ";
    xSize.dump();

    auto resizedX = rewriter.create<ResizeOp>(
        op.getLoc(), resizedXType, subX, xSize,
        op->template getAttrOfType<mlir::BoolAttr>("align_corners"),
        op->template getAttrOfType<mlir::BoolAttr>("half_pixel_centers"));

    // DEBUG
    llvm::dbgs() << "resizedX: ";
    resizedX.dump();

    resizedXs.push_back(resizedX);
  }

  for (auto i = numDims - 1; i >= 0; --i) {
    auto numSplit = numSplits[i];
    if (numSplit == 1)
      continue;
    auto dimSize = xShapeVec[i];
    auto curResizedXs = decltype(resizedXs)();
    for (auto j = 0; j < resizedXs.size(); j += numSplit) {
      auto concatXs = std::vector<mlir::Value>(
          resizedXs.begin() + j, resizedXs.begin() + j + numSplit);
      auto curResizedX = Concat(rewriter, op.getLoc(), concatXs, i);

      // DEBUG
      llvm::dbgs() << "curResizedX: ";
      curResizedX.dump();

      curResizedXs.push_back(curResizedX);
    }
    resizedXs = std::move(curResizedXs);
  }

  // DEBUG
  llvm::dbgs() << "number of resized xs left: " << resizedXs.size() << "\n";

  rewriter.replaceOp(op, resizedXs);

  return mlir::success();
}

mlir::Value PiecewiseRegression(
    mlir::PatternRewriter &rewriter, const mlir::Location &loc,
    const mlir::Value &x, const CalibrationData::Coeffs &coeffs);


void AddMyTFPass(llvm::StringRef modelDir, mlir::OpPassManager *passManager);

void AddMyTFLPass(llvm::StringRef modelDir, mlir::OpPassManager *passManager);
} // namespace MyPass

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_TF_TFL_PASSES_H_
