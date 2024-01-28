GPU_ID=0
DATASET=CUB200
WLUL=2.5
python src/main.py  --root ${DATASET} --batch_size 24 --logdir vis/ --gpu_id ${GPU_ID} --backbone resnet50 --label_ratio 15 --pretrained --mem_use_confid --w_pfa_l ${WLUL} --w_pfa_ul ${WLUL} --momentum_proto 0.7
python src/main.py  --root ${DATASET} --batch_size 24 --logdir vis/ --gpu_id ${GPU_ID} --backbone resnet50 --label_ratio 30 --pretrained --mem_use_confid --w_pfa_l ${WLUL} --w_pfa_ul ${WLUL} --momentum_proto 0.7
python src/main.py  --root ${DATASET} --batch_size 24 --logdir vis/ --gpu_id ${GPU_ID} --backbone resnet50 --label_ratio 50 --pretrained --mem_use_confid --w_pfa_l ${WLUL} --w_pfa_ul ${WLUL} --momentum_proto 0.7
