from model_adapter import get_decoder_layers, get_named_linear, set_named_linear
from qr_core_ver2 import QuotRemLinearV2, IntActLinear


# 데이터 분석 (results_qr_stats) 기반 모듈 분류:
# - QR: outlier가 의미 있어 QR이 INT 대비 7~30% MSE 개선되는 모듈
# - INT: fallback ~99%이고 QR/INT MSE 비율 ~1.0인 모듈 (QR 무용)
ARCH_QR_DEFAULTS = {
    "opt":   ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "fc1"],
    "llama": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "mlp.down_proj"],
}


def replace_modules_v2(
    model,
    arch,
    layer_indices,
    target_modules,
    qr_modules,
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
    # INT activation params (QR이 아닌 모듈용)
    int_act_bits,
    int_act_group_size,
):
    """
    target_modules 중 qr_modules에 속하면 QuotRemLinearV2로,
    아니면 IntActLinear로 교체한다.
    """
    layers = get_decoder_layers(model, arch)
    replaced = []  # [(name, kind)] 형태로 어떤 모듈이 어떤 클래스로 교체됐는지 기록

    qr_set = set(qr_modules)

    for layer_idx in layer_indices:
        layer = layers[layer_idx]
        for module_name in target_modules:
            old_module = get_named_linear(layer, module_name)
            full_name = f"layer{layer_idx}.{module_name}"

            if module_name in qr_set:
                new_module = QuotRemLinearV2(
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
                    residual_clip_alpha=residual_clip_alpha,
                    debug_name=full_name,
                )
                kind = "QR"
            else:
                new_module = IntActLinear(
                    base_linear=old_module,
                    enable_weight_quant=enable_weight_quant,
                    weight_bits=weight_bits,
                    weight_quant_mode=weight_quant_mode,
                    weight_ch_axis=weight_ch_axis,
                    weight_group_size=weight_group_size,
                    weight_scale_method=weight_scale_method,
                    weight_scale_shrink_factors=weight_scale_shrink_factors,
                    act_bits=int_act_bits,
                    act_group_size=int_act_group_size,
                    debug_name=full_name,
                )
                kind = "INT"

            set_named_linear(layer, module_name, new_module)
            replaced.append((full_name, kind))

    return replaced


def get_default_qr_modules(arch):
    if arch not in ARCH_QR_DEFAULTS:
        raise ValueError(f"Unknown arch: '{arch}'. Available: {list(ARCH_QR_DEFAULTS.keys())}")
    return list(ARCH_QR_DEFAULTS[arch])
