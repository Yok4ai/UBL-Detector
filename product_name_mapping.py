"""
Product Name Mapping

This module contains mappings between:
1. data.yaml class names -> human-readable product names
2. Default size variants for products that commonly appear in single sizes

The reverse mapping is used to display actual product names instead of class names
in detection outputs.
"""

# Reverse mapping: class_name -> default product name with size
# Format: 'class_name': 'PRODUCT NAME WITH SIZE'
CLASS_TO_PRODUCT_NAME = {
    # Vaseline - Winter Lotions
    'vaseline_hw': 'VASELINE HW 300ML',
    'vaseline_tm': 'VASELINE TM 300ML',
    'vaseline_gluta_rad': 'VASELINE GLUTA-HYA DEWY RADIANCE 200ML',
    'vaseline_gluta_flawless': 'VASELINE GLUTA-HYA FLAWLESS GLOW 200ML',
    'vaseline_aloe': 'VASELINE ALOE 200ML',
    
    # Oral Care
    'pepsodent_advanced_salt': 'PEPSODENT ADVANCED SALT 140G',
    'closeup_lemon_salt': 'CLOSEUP LEMON & SEA SALT 140G',
    'pepsodent_germicheck': 'PEPSODENT GERMICHECK 140G',
    'pepsodent_sensitive_expert': 'PEPSODENT SENSITIVE EXPERT 140G',
    
    # Lux Bodywash
    'lux_blk_orchd': 'LUX BODYWASH BLACK ORCHID 245ML',
    'lux_brightening_vitamin': 'LUX BODYWASH BRIGHTENING VITAMIN 245ML',
    'lux_freeasia_scnt': 'LUX BODYWASH FREESIA SCENT 245ML',
    'lux_french_rose': 'LUX BODYWASH FRENCH ROSE 245ML',
    
    # Nutrition Store - Horlicks
    'horlicks_junior': 'JUNIOR HORLICKS 500G JAR STAGE 2',
    'horlicks_junior_s1': 'JUNIOR HORLICKS 500G JAR STAGE 1',
    'horlicks_mother': 'MOTHER HORLICKS BIB 350G',
    'horlicks_std': 'STANDARD HORLICKS 500G JAR',  # Default to 500G (most common)
    'horlicks_choco': 'CHOCOLATE HORLICKS JAR 500G',
    'horlicks_women': 'HORLICKS WOMEN JAR 400G',
    'horlicks_lite': 'HORLICKS LITE JAR 330G',
    'maltova_std': 'MALTOVA BIB 400G',
    'boost_std': 'BOOST HFD STANDARD JAR 400G',
    
    # Clear Shampoos
    'clear_csm_small': 'CLEAR MALE SHAMPOO CSM 180ML',
    'clear_csm_large': 'CLEAR MALE SHAMPOO CSM 450ML',
    'clear_ahf': 'CLEAR SHAMPOO ANTI HAIR FALL 180ML',
    'clear_cac': 'CLEAR SHAMPOO COMPLETE ACTIVE CARE 180ML',
    
    # Dove Products
    'dove_cond': 'DOVE HAIR RINSE OUT CONDITIONER 180ML',
    'dove_hg': 'DOVE HEALTHY GROWTH 170ML',
    'dove_hfr_small': 'DOVE SHAMPOO HAIR FALL RESCUE 170ML',
    'dove_hfr_large': 'DOVE SHAMPOO HAIR FALL RESCUE 650ML',
    'dove_irp_small': 'DOVE SHAMPOO INTENSIVE REPAIR 170ML',
    'dove_irp_large': 'DOVE SHAMPOO IRP 650ML',
    'dove_no': 'DOVE SHAMPOO NOURISHING OIL 170ML',
    'dove_oxg': 'DOVE SMP OXG MT 170ML',
    'dove_mask_25': 'DOVE HAIR MASK 25ML',
    'dove_nr_lotion': 'DOVE NOURISHING LOTION',
    
    # Glow & Lovely
    'gl_foundation_crm': 'GLOW & LOVELY BALM FOUNDATION CRM 40G',
    'gl_aryuvedic_crm': 'GLOW & LOVELY FACIAL MOIST AYURVEDIC 50G',
    'gl_mltvit_crm': 'GLOW & LOVELY FACIAL MST MULTIVIT CRM 50G',
    'gl_insta_glow_fw': 'GLOW & LOVELY FACE WASH INSTA GLOW 100G',
    'gl_sunscrn_crm': 'GLOW & LOVELY SUNSCREEN CREAM 50G',
    
    # Ponds
    'ponds_white_beauty_crm': 'PONDS WHITE BEAUTY CRM 35G',
    'ponds_pure_white_fw': 'PONDS PURE WHITE FACE WASH 100G',
    'ponds_oil_control_fw': 'PONDS FACE WASH OIL CONTROL 100ML',
    'ponds_pure_white_clay_fw': 'PONDS PURE WHITE CLAY FOAM 90G',
    'ponds_white_beauty_clay_fw': 'PONDS WHITE BEAUTY CLAY FOAM 90G',
    'ponds_white_beauty_fw': 'PONDS WHITE BEAUTY FACE WASH 100G',
    
    # Sunsilk Shampoos
    'sunsilk_black_small': 'SUNSILK SHAMPOO BLACK 180ML',
    'sunsilk_black_large': 'SUNSILK SHAMPOO BLACK 450ML',
    'sunsilk_fresh': 'SUNSILK SHAMPOO FRESHNESS 195ML',
    'sunsilk_hfs': 'SUNSILK SHAMPOO HFS 340ML',
    'sunsilk_hfs_rinse': 'SUNSILK HFS RINSE OUT CONDITIONER 170',
    'sunsilk_hrr': 'SUNSILK SHAMPOO HIJAB RECHARGE REFRESH 180ML',
    'sunsilk_tl_small': 'SUNSILK SHAMPOO THICK & LONG 375ML',
    'sunsilk_tl_large': 'SUNSILK SHAMPOO THICK & LONG 450ML',
    'sunsilk_volume': 'SUNSILK SHAMPOO VOLUME 195ML',
    'sunsilk_onion': 'SUNSILK SHAMPOO ONION 375ML',
    'sunsilk_serum_25': 'SUNSILK SERUM 25ML',
    
    # Tresemme
    'tresemme_cr': 'TRESEMME SHAMPOO COLOR REVITALISE 580ML',
    'tresemme_ks_small': 'TRESEMME SHAMPOO KERATIN SMOOTH 340ML',
    'tresemme_ks_large': 'TRESEMME SHAMPOO KERATIN SMOOTH 580ML',
    'tresemme_ks_white': 'TRESEMME CONDITIONER KERATIN SMOOTH 190ML',
    'tresemme_mask_25': 'TRESEMME HAIR MASK 25ML',
    'tresemme_serum_25': 'TRESEMME SERUM 25ML',
    'treseme_sampoo_bond_plex': 'TRESEMME SHAMPOO BOND PLEX',
    
    # Lifebuoy
    'lifebuoy_pump': 'LIFEBUOY HANDWASH PUMP',
    'lifebuoy_refill_pouch': 'LIFEBUOY HANDWASH REFILL POUCH',
    
    # Surf Excel
    'surf_excel_matic_1l': 'SURF EXCEL MATIC 1L',
    'surf_excel_matic_500ml': 'SURF EXCEL MATIC 500ML',
    
    # Vim
    'vim_liquid_500ml': 'VIM LIQUID 500ML',
}


# Multi-size products with their variants
# Format: 'class_name': [list of sizes in ascending order]
PRODUCT_SIZE_VARIANTS_CONFIG = {
    # Horlicks - Standard has both 500g and 1kg
    'horlicks_std': ['500g', '1kg'],
    
    # Other Horlicks variants (single size)
    'horlicks_junior': ['500g'],
    'horlicks_junior_s1': ['500g'],
    'horlicks_mother': ['350g'],
    'horlicks_choco': ['500g'],
    'horlicks_women': ['400g'],
    'horlicks_lite': ['330g'],
    
    # Maltova & Boost
    'maltova_std': ['400g'],
    'boost_std': ['400g'],
    
    # Vaseline (mostly single size, but some have 200ml and 300ml)
    'vaseline_hw': ['200ml', '300ml'],
    'vaseline_tm': ['200ml', '300ml'],
    'vaseline_aloe': ['200ml', '300ml'],
    'vaseline_gluta_rad': ['200ml'],
    'vaseline_gluta_flawless': ['200ml'],
    
    # Oral Care
    'pepsodent_advanced_salt': ['140g'],
    'pepsodent_germicheck': ['140g'],
    'pepsodent_sensitive_expert': ['140g'],
    'closeup_lemon_salt': ['140g'],
    
    # Lux Bodywash
    'lux_blk_orchd': ['245ml'],
    'lux_brightening_vitamin': ['245ml'],
    'lux_freeasia_scnt': ['245ml'],
    'lux_french_rose': ['245ml'],
    
    # Clear Shampoo already has small/large distinction
    'clear_csm_small': ['180ml', '330ml'],
    'clear_csm_large': ['450ml'],
    'clear_ahf': ['180ml', '350ml'],
    'clear_cac': ['180ml', '350ml'],
    
    # Dove
    'dove_hfr_small': ['170ml', '340ml'],
    'dove_hfr_large': ['650ml'],
    'dove_irp_small': ['170ml', '340ml'],
    'dove_irp_large': ['650ml'],
    'dove_no': ['170ml', '340ml'],
    'dove_hg': ['170ml', '340ml'],
    'dove_oxg': ['170ml', '340ml'],
    'dove_cond': ['180ml'],
    'dove_mask_25': ['25ml'],
    'dove_nr_lotion': ['200ml'],
    
    # Glow & Lovely
    'gl_foundation_crm': ['40g'],
    'gl_aryuvedic_crm': ['50g'],
    'gl_mltvit_crm': ['50g', '100g'],
    'gl_insta_glow_fw': ['100g'],
    'gl_sunscrn_crm': ['50g'],
    
    # Ponds
    'ponds_white_beauty_crm': ['25g', '35g'],
    'ponds_pure_white_fw': ['100g'],
    'ponds_oil_control_fw': ['100ml'],
    'ponds_pure_white_clay_fw': ['90g'],
    'ponds_white_beauty_clay_fw': ['90g'],
    'ponds_white_beauty_fw': ['100g'],
    
    # Sunsilk
    'sunsilk_black_small': ['180ml', '375ml'],
    'sunsilk_black_large': ['450ml'],
    'sunsilk_tl_small': ['180ml', '340ml'],
    'sunsilk_tl_large': ['450ml'],
    'sunsilk_fresh': ['195ml', '375ml'],
    'sunsilk_hfs': ['180ml', '340ml'],
    'sunsilk_hfs_rinse': ['170ml'],
    'sunsilk_hrr': ['180ml', '375ml'],
    'sunsilk_volume': ['195ml', '375ml'],
    'sunsilk_serum_25': ['25ml'],
    'sunsilk_onion': ['375ml'],
    
    # Tresemme
    'tresemme_ks_small': ['340ml'],
    'tresemme_ks_large': ['580ml'],
    'tresemme_cr': ['580ml'],
    'tresemme_ks_white': ['190ml'],
    'tresemme_mask_25': ['25ml'],
    'tresemme_serum_25': ['25ml'],
    'treseme_sampoo_bond_plex': ['580ml'],
    
    # Lifebuoy
    'lifebuoy_pump': ['250ml'],
    'lifebuoy_refill_pouch': ['185ml'],
    
    # Surf Excel
    'surf_excel_matic_500ml': ['500ml'],
    'surf_excel_matic_1l': ['1l'],
    
    # Vim
    'vim_liquid_500ml': ['500ml'],
}


def get_product_display_name(class_name: str, size_variant: str = None) -> str:
    """
    Get the human-readable product name for a class.
    
    Args:
        class_name: The data.yaml class name (e.g., 'horlicks_std')
        size_variant: Optional size variant (e.g., '1kg', '500g')
    
    Returns:
        Human-readable product name with size
    """
    # Get base product name
    base_name = CLASS_TO_PRODUCT_NAME.get(class_name, class_name.upper())
    
    # If size variant is provided and differs from default, update it
    if size_variant and size_variant != 'N/A':
        # For products like horlicks_std that have multiple sizes
        if class_name in PRODUCT_SIZE_VARIANTS_CONFIG:
            variants = PRODUCT_SIZE_VARIANTS_CONFIG[class_name]
            if len(variants) > 1 and size_variant in variants:
                # Replace the size in the base name
                # Example: "STANDARD HORLICKS 500G JAR" -> "STANDARD HORLICKS 1KG JAR"
                for variant in variants:
                    base_name = base_name.replace(variant.upper(), size_variant.upper())
                    
    return base_name


def get_default_size(class_name: str) -> str:
    """
    Get the default size for a product class.
    
    Args:
        class_name: The data.yaml class name
    
    Returns:
        Default size string (e.g., '500g', '300ml')
    """
    variants = PRODUCT_SIZE_VARIANTS_CONFIG.get(class_name, [])
    if not variants:
        return 'N/A'
    
    # For products with multiple sizes, default to the most common one
    # which is typically the smaller size (index 0) unless specified otherwise
    
    # Nutrition Store - Default to most common sizes
    if class_name == 'horlicks_std':
        return '500g'  # Most common (smaller jar)
    
    # Winter Lotion - Default to larger 300ml bottles
    elif class_name in ['vaseline_hw', 'vaseline_tm', 'vaseline_aloe']:
        return '300ml'  # Most common (larger bottle)
    
    # Sunsilk - Default to larger bottles
    elif class_name == 'sunsilk_tl_small':
        return '340ml'  # Most common (larger size)
    elif class_name == 'sunsilk_hfs':
        return '340ml'  # Most common (larger size)
    elif class_name in ['sunsilk_black_small', 'sunsilk_fresh', 'sunsilk_hrr', 'sunsilk_volume']:
        return '375ml'  # Most common (larger size)
    
    # Dove - Default to larger bottles where applicable
    elif class_name in ['dove_hfr_small', 'dove_irp_small', 'dove_no', 'dove_hg', 'dove_oxg']:
        return '340ml'  # Most common (larger bottle)
    
    # Clear - Default to larger bottles
    elif class_name in ['clear_csm_small', 'clear_ahf', 'clear_cac']:
        return '350ml' if class_name != 'clear_csm_small' else '330ml'  # Most common (larger bottle)
    
    # Ponds - Default to larger size
    elif class_name == 'ponds_white_beauty_crm':
        return '35g'  # Most common (larger size)
    elif class_name == 'gl_mltvit_crm':
        return '100g'  # Most common (larger size)
    
    return variants[0]  # Default to first (smallest) size for others
