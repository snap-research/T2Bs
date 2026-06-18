log=0000
idname=monkey
neutral_expression=eyes_half_mouth_open_wide

# # generate meshes from Step1X-3D
cd Step1X-3D
python3 inference.py --animal ${idname}
cd ..


# # registration
python3 train.py --idname ${idname} --n_views 25 --log ${log} --deform_fc --normalize_mesh --view_independent --neutral ${neutral_expression} --use_loss_n


log=0000
neutral_expression=halfo_m_o_e

idname=Aardvark_Plastic_Toy_Render_mohawk_fur_a_Såanta_hat____a_chef_hat_1
python3 train.py --idname ${idname} --n_views 25 --log 0001 --deform_fc --normalize_mesh --view_independent --neutral ${neutral_expression} --use_loss_n


idname=Antelope_Plastic_Toy_Render_round_face_a_headband____a_superhero_mask_1
python3 train.py --idname ${idname} --n_views 25 --log 0001 --deform_fc --normalize_mesh --view_independent --neutral ${neutral_expression} --use_loss_n


neutral_expression=halfo_m_o_e
idname=Akita_Dog_DreamWorks_Style_3D_long_snout_a_crown_of_leaves____a_wizard_hat_1
python3 train.py --idname ${idname} --n_views 25 --log 0008 --deform_fc --normalize_mesh --view_independent --neutral ${neutral_expression}

neutral_expression=halfo_m_o_e
idname=Akita_Dog_DreamWorks_Style_3D_long_snout_a_crown_of_leaves____a_wizard_hat_1
python3 train.py --idname ${idname} --n_views 25 --log 0029 --normalize_mesh --neutral ${neutral_expression}


neutral_expression=halfo_m_o_e
idname=Aardvark_Plastic_Toy_Render_mohawk_fur_a_Såanta_hat____a_chef_hat_1
python3 train.py --idname ${idname} --n_views 25 --log 0024 --normalize_mesh --neutral ${neutral_expression}

