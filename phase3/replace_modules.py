from model_adapter import get_decoder_layers, get_named_linear, set_named_linear
from qr_core import QuotRemLinear


def replace_modules_with_quotrem_linear(
    model,
    arch,
    layer_indices,
    module_names,
    enable_weight_quant,
    weight_bits,
    weight_quant_mode,
    weight_ch_axis,
    weight_group_size,
    weight_scale_method,
    weight_scale_shrink_factors,
    q_bits,
    r_bits,
    base_group_size,
    r_group_size,
    selective_base_threshold,
    module_selective_base_thresholds,
    selective_int_bits,
    residual_clip_alpha,
    collect_residuals=False,
):
    layers = get_decoder_layers(model, arch)
    replaced_names = []

    for layer_idx in layer_indices:
        layer = layers[layer_idx]
        for module_name in module_names:
            old_module = get_named_linear(layer, module_name)
            # module별 threshold가 지정된 경우 해당 module만 덮어쓰고,
            # 지정되지 않은 module은 전역 threshold를 그대로 사용한다.
            module_threshold = module_selective_base_thresholds.get(
                module_name, selective_base_threshold
            )
            new_module = QuotRemLinear(
                base_linear=old_module,
                enable_weight_quant=enable_weight_quant,
                weight_bits=weight_bits,
                weight_quant_mode=weight_quant_mode,
                weight_ch_axis=weight_ch_axis,
                weight_group_size=weight_group_size,
                weight_scale_method=weight_scale_method,
                weight_scale_shrink_factors=weight_scale_shrink_factors,
                q_bits=q_bits,
                r_bits=r_bits,
                base_group_size=base_group_size,
                r_group_size=r_group_size,
                selective_base_threshold=module_threshold,
                selective_int_bits=selective_int_bits,
                residual_clip_alpha=residual_clip_alpha,
                collect_residuals=collect_residuals,
                debug_name=f"layer{layer_idx}.{module_name}",
            )
            set_named_linear(layer, module_name, new_module)
            replaced_names.append(f"layer{layer_idx}.{module_name}")

    return replaced_names
