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
#include "tensorflow/core/public/session.h"
#include "tensorflow/lite/toco/model_flags.pb.h"
#include "tensorflow/lite/toco/toco_flags.pb.h"
// EDIT
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
// variables.
void AddTFToTFLConversionPasses(const toco::ModelFlags& model_flags,
                                const toco::TocoFlags& toco_flags,
                                const mlir::TFL::PassConfig& pass_config,
                                mlir::OpPassManager* pass_manager,
                                llvm::Optional<tensorflow::Session*> session);

void AddTFToTFLConversionPasses(const mlir::TFL::PassConfig& pass_config,
                                mlir::OpPassManager* pass_manager,
                                llvm::Optional<tensorflow::Session*> session);

// Add the Quantization passes, specified in the quant_specs, into a pass
// manager.
void AddQuantizationPasses(const mlir::TFL::QuantizationSpecs& quant_specs,
                           mlir::OpPassManager* pass_manager);

}  // namespace tensorflow

namespace MyPass {
class CalibrationData {
private:
  constexpr static const char *coeffCalculatorPath_ =
      "/home/robert/compiler-lab/edgetpu-pass/src/tools/"
      "poly_approx_get_coeff.py";
  std::string path_;
  std::unordered_map<std::string, std::pair<float, float>> data_;

public:
  explicit CalibrationData(
      const std::string &path, const std::string &filename);

  std::pair<float, float> getRange(mlir::Location loc) const;

  void getPolyCoeff(
      mlir::Location loc, const std::string &function,
      const std::string &method, int n, float coeffs[]) const;
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

struct ReplaceExpOp : public mlir::OpRewritePattern<mlir::TFL::ExpOp> {
  explicit ReplaceExpOp(mlir::MLIRContext *ctx);

  mlir::LogicalResult matchAndRewrite(
      mlir::TFL::ExpOp op, mlir::PatternRewriter &rewriter) const override;
};

struct ReplaceLogOp : public mlir::OpRewritePattern<mlir::TFL::LogOp> {
  explicit ReplaceLogOp(mlir::MLIRContext *ctx);

  mlir::LogicalResult matchAndRewrite(
      mlir::TFL::LogOp op, mlir::PatternRewriter &rewriter) const override;
};

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

struct ReplaceAveragePool2DOp : public mlir::OpRewritePattern<
    mlir::TFL::AveragePool2DOp> {
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

struct AllTFLPasses : public mlir::PassWrapper<AllTFLPasses,
    mlir::OperationPass<>> {
  void runOnOperation() override;
};

struct ReplaceSoftplusOp : public mlir::OpRewritePattern<mlir::TF::SoftplusOp> {
  CalibrationData calibrationData_;

  ReplaceSoftplusOp(mlir::MLIRContext *ctx, const std::string &savedModelPath);

  mlir::LogicalResult matchAndRewrite(
      mlir::TF::SoftplusOp op, mlir::PatternRewriter &rewriter) const override;
};

struct ReplaceTFUnpackOp : public mlir::OpRewritePattern<mlir::TF::UnpackOp> {
  explicit ReplaceTFUnpackOp(mlir::MLIRContext *ctx);

  mlir::LogicalResult matchAndRewrite(
      mlir::TF::UnpackOp op, mlir::PatternRewriter &rewriter) const override;
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
      mlir::TFL::MaxPool2DOp op, mlir::PatternRewriter &rewriter) const override;
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
}

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_TF_TFL_PASSES_H_
