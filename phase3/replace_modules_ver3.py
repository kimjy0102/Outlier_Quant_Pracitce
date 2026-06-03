from model_adapter import get_decoder_layers, get_named_linear, set_named_linear
from qr_core_ver3 import QuotRemLinearV3


def replace_modules_v3(
    model,
    arch,
    layer_indices,
    target_modules,
    calib_modules,                       # artifact["modules"]: dict {(layer, name): {perm, per_group_max_abs, ...}}
    selective_base_threshold,            # 전역 threshold (ver1과 동일 시맨틱)
    module_selective_base_thresholds,    # dict[module_name] = float (module별 override)
    # weight quant
    enable_weight_quant,
    weight_bits,
    weight_quant_mode,
    weight_ch_axis,
    weight_group_size,
    weight_scale_method,
    weight_scale_shrink_factors,
    # QR params
    q_bits,
    r_bits,
    base_group_size,
    r_group_size,
    residual_clip_alpha,
    # INT4 (mask=False group) params
    act_bits,
    act_group_size,
    # Phase 1: base / INT scale 정적화
    use_static_scales=False,
):
    layers = get_decoder_layers(model, arch)
    replaced = []   # (full_name, threshold_used, qr_groups, total_groups)

    for layer_idx in layer_indices:
        layer = layers[layer_idx]
        for module_name in target_modules:
            key = (layer_idx, module_name)
            if key not in calib_modules:
                raise KeyError(
                    f"Calibration artifact missing entry for {key}. "
                    f"Re-run calibration with matching target_modules."
                )
            entry = calib_modules[key]
            perm              = entry["perm"]
            per_group_max_abs = entry["per_group_max_abs"]

            module_threshold = module_selective_base_thresholds.get(
                module_name, selective_base_threshold
            )

            old_module = get_named_linear(layer, module_name)
            new_module = QuotRemLinearV3(
                base_linear=old_module,
                perm=perm,
                per_group_max_abs=per_group_max_abs,
                threshold=module_threshold,
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
                residual_clip_alpha=residual_clip_alpha,
                act_bits=act_bits,
                act_group_size=act_group_size,
                use_static_scales=use_static_scales,
                debug_name=f"layer{layer_idx}.{module_name}",
            )

            qr_groups    = int(new_module.group_mask.sum().item())
            total_groups = int(new_module.group_mask.numel())
            replaced.append((
                f"layer{layer_idx}.{module_name}",
                module_threshold,
                qr_groups,
                total_groups,
            ))
            set_named_linear(layer, module_name, new_module)

    return replaced
