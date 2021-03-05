#pragma once
#include <torch/extension.h>

using namespace at;


std::tuple<Tensor, Tensor> ctc_loss_gpu(
    const Tensor& log_probs, 
    const Tensor& targets, 
    IntArrayRef input_lengths, IntArrayRef target_lengths,
    int64_t BLANK, bool zero_infinity);



Tensor ctc_loss_backward_gpu(
    const Tensor& grad, 
    const Tensor& log_probs, 
    const Tensor& targets,
    IntArrayRef input_lengths, IntArrayRef target_lengths,
    const Tensor& neg_log_likelihood, const Tensor& log_alpha, 
    int64_t BLANK, bool zero_infinity);
