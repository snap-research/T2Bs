log=0000

# idname=cow
# neutral_expression=eyes_half_mouth_open_wide
# python3 train.py --idname ${idname} --n_views 25 --log ${log} --deform_fc --normalize_mesh --view_independent --neutral ${neutral_expression} --use_loss_n

# idname=bear
# neutral_expression=eyes_half_mouth_open_wide
# python3 train.py --idname ${idname} --n_views 25 --log ${log} --deform_fc --normalize_mesh --view_independent --neutral ${neutral_expression} --use_loss_n

# idname=crying_cat
# neutral_expression=mouth_open
# python3 train.py --idname ${idname} --n_views 25 --log ${log} --deform_fc --normalize_mesh --view_independent --neutral ${neutral_expression} --use_loss_n

idname=dog
neutral_expression=mouth_open_wide
python3 train.py --idname ${idname} --n_views 25 --log ${log} --deform_fc --normalize_mesh --view_independent --neutral ${neutral_expression} --use_loss_n

# idname=fox
# neutral_expression=eyes_half_mouth_open_wide
# python3 train.py --idname ${idname} --n_views 25 --log ${log} --deform_fc --normalize_mesh --view_independent --neutral ${neutral_expression} --use_loss_n

# idname=moon
# neutral_expression=mouth_open
# python3 train.py --idname ${idname} --n_views 25 --log ${log} --deform_fc --normalize_mesh --view_independent --neutral ${neutral_expression} --use_loss_n
