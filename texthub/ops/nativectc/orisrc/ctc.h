#pragma once

#ifdef WITH_CUDA
#include "cuda/ctc.h"
#endif

std::tuple<at::Tensor, at::Tensor> nativectc_forward(
     at::Tensor& log_probs,  at::Tensor& targets,
     at::IntArrayRef input_lengths, at::IntArrayRef target_lengths,
    int64_t BLANK, bool zero_infinity
) {
    if(log_probs.device().type()==at::DeviceType::CUDA){
            #ifdef WITH_CUDA
    return ctc_loss_gpu(
        log_probs, targets, input_lengths, target_lengths, BLANK,zero_infinity
    );
#else
    AT_ERROR("Not compiled with GPU support");
#endif
    }
    // if (log_probs.type().is_cuda()) {

    // }
    AT_ERROR("Not implemented on the CPU");
}


at::Tensor nativectc_backward(
      at::Tensor grad,
     at::Tensor log_probs,
     at::Tensor targets,
    at::IntArrayRef input_lengths,
    at::IntArrayRef target_lengths,
     at::Tensor neg_log_likelihood,
     at::Tensor log_alpha,
    int64_t BLANK,
    bool zero_infinity
) {
    if(log_probs.device().type()==at::DeviceType::CUDA){
#ifdef WITH_CUDA
    return ctc_loss_backward_gpu(
        grad,
        log_probs, targets, input_lengths, target_lengths,
        neg_log_likelihood, log_alpha,
        BLANK,zero_infinity
    );
#else
    AT_ERROR("Not compiled with GPU support");
#endif
    }
    AT_ERROR("Not implemented on the CPU");
}