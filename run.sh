#python3 baseline.py -l btl -av 2;
#python3 baseline.py -l btl -av 3;
#python3 baseline.py -l btl -av 4;
#python3 baseline.py -l btl -av 5 -wd 1e-3;
#python3 baseline.py -l btl -av 6 -wd 1e-3;
#python3 baseline.py -l btl -av 7 -wd 1e-3;
#python3 baseline.py -l btl -av 8 -wd 1e-3;

#python3 baseline.py -l focal -av 2 -m tf_efficientnet_b7_ns -wd 1e-4 --amp True;
#python3 baseline.py -l focal -av 5 -m tf_efficientnet_b7_ns -wd 1e-4 --amp True;
#python3 baseline.py -l focal -av 21 -m tf_efficientnet_b7_ns -wd 1e-4 --amp True;

#python3 baseline.py -l btl -av 9 -wd 1e-3;
#python3 baseline.py -l btl -av 10 -wd 1e-3;
#python3 baseline.py -l btl -av 11 -wd 1e-3;

#python3 baseline.py -av 22;
#python3 baseline.py -av 23;

#python3 baseline.py -av 24 -m tf_efficientnetv2_m_in21ft1k --amp True -mu True;

#python3 baseline.py -av 24 -bs 32 -m tf_efficientnetv2_l_in21ft1k --amp True;
#python3 baseline.py -av 24 -bs 64 -m tf_efficientnetv2_m_in21ft1k --amp True;

#python3 baseline.py -av 24 -m tf_efficientnetv2_m_in21ft1k --amp True;
#python3 baseline.py -av 24 -m tf_efficientnetv2_m_in21ft1k --amp True;

#python3 baseline.py -av 25 -bs 32 -is 224 -m convnext_xlarge_in22ft1k --amp True;
#python3 baseline.py -av 26 -bs 16 -is 384 -m beit_large_patch16_384 --amp True;

#python3 baseline.py -av 24 -bs 48 -dp new_data -m swin_base_patch4_window12_384_in22k --amp True;

#python3 baseline.py -av 24 -l arcface -bs 48 -m dolg --amp True;
#python3 baseline.py -av 24 -l arcface -bs 48 -m convnext_large_384_in22ft1k --amp True;


#python3 baseline.py -av 24 -bs 28 -m arc_swin_base_patch4_window12_384_in22k -l arcface --amp True;
python3 baseline.py -av 24 -bs 48 -m arc_convnext_large_384_in22ft1k -l arcface --amp True;