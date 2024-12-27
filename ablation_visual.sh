# nohup ./ablation_visual.sh > ablation_visual.log 2>&1 &

python train.py --experiment_dir=./exper_visual   --sf=True    --sftransform=True  --dataset=mvtecad       --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/mvtec_anomaly_detection       --classname=carpet          --know_class=None --nAnomaly=1 &&
python train.py --experiment_dir=./exper_visual   --sf=True    --sftransform=True  --dataset=mvtecad       --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/mvtec_anomaly_detection       --classname=carpet          --know_class=None --nAnomaly=1 &&
python train.py --experiment_dir=./exper_visual   --sf=True    --sftransform=True  --dataset=mvtecad       --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/mvtec_anomaly_detection       --classname=carpet          --know_class=None --nAnomaly=1 &&

python train.py --experiment_dir=./exper_visual   --sf=True    --sftransform=True   --dataset=AITEX         --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/AITEX_anomaly_detection       --classname=AITEX           --know_class=None --nAnomaly=1 &&
python train.py --experiment_dir=./exper_visual   --sf=True    --sftransform=True   --dataset=AITEX         --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/AITEX_anomaly_detection       --classname=AITEX           --know_class=None --nAnomaly=1 &&
python train.py --experiment_dir=./exper_visual   --sf=True    --sftransform=True   --dataset=AITEX         --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/AITEX_anomaly_detection       --classname=AITEX           --know_class=None --nAnomaly=1 &&

python train.py --experiment_dir=./exper_visual   --sf=True    --sftransform=True   --dataset=elpv          --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/elpv_anomaly_detection        --classname=elpv            --know_class=None --nAnomaly=1 &&
python train.py --experiment_dir=./exper_visual   --sf=True    --sftransform=True   --dataset=elpv          --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/elpv_anomaly_detection        --classname=elpv            --know_class=None --nAnomaly=1 &&
python train.py --experiment_dir=./exper_visual   --sf=True    --sftransform=True   --dataset=elpv          --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/elpv_anomaly_detection        --classname=elpv            --know_class=None --nAnomaly=1 &&

python train.py --experiment_dir=./exper_visual   --sf=True    --sftransform=True   --dataset=hyperkvasir   --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/hyperkvasir_anomaly_detection --classname=hyperkvasir     --know_class=None --nAnomaly=1      --outlier_root=/root/autodl-tmp/ad_wxf/ad_dataset/HeadCT_anomaly_detection &&
python train.py --experiment_dir=./exper_visual   --sf=True    --sftransform=True   --dataset=hyperkvasir   --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/hyperkvasir_anomaly_detection --classname=hyperkvasir     --know_class=None --nAnomaly=1      --outlier_root=/root/autodl-tmp/ad_wxf/ad_dataset/HeadCT_anomaly_detection &&
python train.py --experiment_dir=./exper_visual   --sf=True    --sftransform=True   --dataset=hyperkvasir   --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/hyperkvasir_anomaly_detection --classname=hyperkvasir     --know_class=None --nAnomaly=1      --outlier_root=/root/autodl-tmp/ad_wxf/ad_dataset/HeadCT_anomaly_detection 
