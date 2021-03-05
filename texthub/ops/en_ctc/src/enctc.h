#pragma once

#ifdef WITH_CUDA
#include "cuda/enctc.h"
#endif

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> enctc_forward(
     at::Tensor& log_probs,  at::Tensor& targets,
     at::IntArrayRef input_lengths, at::IntArrayRef target_lengths,
    int64_t BLANK, bool zero_infinity,float ent_rate
) {
    if(log_probs.device().type()==at::DeviceType::CUDA){
            #ifdef WITH_CUDA
    return enctc_loss_gpu(
        log_probs, targets, input_lengths, target_lengths, BLANK,zero_infinity,ent_rate
    );
#else
    AT_ERROR("Not compiled with GPU support");
#endif
    }
    // if (log_probs.type().is_cuda()) {

    // }
    AT_ERROR("Not implemented on the CPU");
}


at::Tensor enctc_backward(
      at::Tensor grad,
     at::Tensor log_probs,
     at::Tensor targets,
    at::IntArrayRef input_lengths,
    at::IntArrayRef target_lengths,
     at::Tensor neg_log_likelihood,
     at::Tensor log_alpha,
     at::Tensor log_ent,
     at::Tensor log_alpha_ent,
    int64_t BLANK,
    bool zero_infinity,
    float ent_rate
) {
    if(log_probs.device().type()==at::DeviceType::CUDA){
#ifdef WITH_CUDA
    return enctc_loss_backward_gpu(
        grad,
        log_probs, targets, input_lengths, target_lengths,
        neg_log_likelihood, log_alpha,log_ent,log_alpha_ent,
        BLANK,zero_infinity,ent_rate
    );
#else
    AT_ERROR("Not compiled with GPU support");
#endif
    }
    AT_ERROR("Not implemented on the CPU");
}