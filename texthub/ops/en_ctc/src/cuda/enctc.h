#pragma once
#include <torch/extension.h>


at::Tensor enctc_loss_backward_gpu(
    const at::Tensor& grad,
    const at::Tensor& log_probs,
    const at::Tensor& targets,
    at::IntArrayRef input_lengths,
    at::IntArrayRef target_lengths,
    const at::Tensor& neg_log_likelihood,
    const at::Tensor& log_alpha,
    const at::Tensor& log_ent,
    const at::Tensor& log_alpha_ent,
    int64_t BLANK,
    bool zero_infinity,
    float ent_rate
);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> enctc_loss_gpu(
    const at::Tensor& log_probs,
    const at::Tensor& targets,
    at::IntArrayRef input_lengths,
    at::IntArrayRef target_lengths,
    int64_t BLANK,
    bool zero_infinity,float ent_rate
);