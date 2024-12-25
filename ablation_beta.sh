# nohup ./ablation_beta.sh > ablation_beta.log 2>&1 &
# python  -m debugpy --listen 1127 --wait-for-client train.py --experiment_dir=./exper_beta0001   --beta=0.001  --dataset=mvtecad       --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/mvtec_anomaly_detection       --classname=carpet          --know_class=None --nAnomaly=1

python train.py --experiment_dir=./exper_beta0001   --beta=0.001  --dataset=mvtecad       --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/mvtec_anomaly_detection       --classname=carpet          --know_class=None --nAnomaly=1 &&
python train.py --experiment_dir=./exper_beta0001   --beta=0.001  --dataset=mvtecad       --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/mvtec_anomaly_detection       --classname=carpet          --know_class=None --nAnomaly=1 &&
python train.py --experiment_dir=./exper_beta0001   --beta=0.001  --dataset=mvtecad       --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/mvtec_anomaly_detection       --classname=carpet          --know_class=None --nAnomaly=1 &&

python train.py --experiment_dir=./exper_beta0001   --beta=0.001   --dataset=AITEX         --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/AITEX_anomaly_detection       --classname=AITEX           --know_class=None --nAnomaly=1 &&
python train.py --experiment_dir=./exper_beta0001   --beta=0.001   --dataset=AITEX         --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/AITEX_anomaly_detection       --classname=AITEX           --know_class=None --nAnomaly=1 &&
python train.py --experiment_dir=./exper_beta0001   --beta=0.001   --dataset=AITEX         --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/AITEX_anomaly_detection       --classname=AITEX           --know_class=None --nAnomaly=1 &&

python train.py --experiment_dir=./exper_beta0001   --beta=0.001   --dataset=elpv          --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/elpv_anomaly_detection        --classname=elpv            --know_class=None --nAnomaly=1 &&
python train.py --experiment_dir=./exper_beta0001   --beta=0.001   --dataset=elpv          --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/elpv_anomaly_detection        --classname=elpv            --know_class=None --nAnomaly=1 &&
python train.py --experiment_dir=./exper_beta0001   --beta=0.001   --dataset=elpv          --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/elpv_anomaly_detection        --classname=elpv            --know_class=None --nAnomaly=1 &&

python train.py --experiment_dir=./exper_beta0001   --beta=0.001   --dataset=hyperkvasir   --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/hyperkvasir_anomaly_detection --classname=hyperkvasir     --know_class=None --nAnomaly=1      --outlier_root=/root/autodl-tmp/ad_wxf/ad_dataset/HeadCT_anomaly_detection &&
python train.py --experiment_dir=./exper_beta0001   --beta=0.001   --dataset=hyperkvasir   --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/hyperkvasir_anomaly_detection --classname=hyperkvasir     --know_class=None --nAnomaly=1      --outlier_root=/root/autodl-tmp/ad_wxf/ad_dataset/HeadCT_anomaly_detection &&
python train.py --experiment_dir=./exper_beta0001   --beta=0.001   --dataset=hyperkvasir   --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/hyperkvasir_anomaly_detection --classname=hyperkvasir     --know_class=None --nAnomaly=1      --outlier_root=/root/autodl-tmp/ad_wxf/ad_dataset/HeadCT_anomaly_detection &&


python train.py --experiment_dir=./exper_beta01   --beta=0.1  --dataset=mvtecad       --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/mvtec_anomaly_detection       --classname=carpet          --know_class=None --nAnomaly=1 &&
python train.py --experiment_dir=./exper_beta01   --beta=0.1  --dataset=mvtecad       --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/mvtec_anomaly_detection       --classname=carpet          --know_class=None --nAnomaly=1 &&
python train.py --experiment_dir=./exper_beta01   --beta=0.1  --dataset=mvtecad       --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/mvtec_anomaly_detection       --classname=carpet          --know_class=None --nAnomaly=1 &&

python train.py --experiment_dir=./exper_beta01   --beta=0.1   --dataset=AITEX         --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/AITEX_anomaly_detection       --classname=AITEX           --know_class=None --nAnomaly=1 &&
python train.py --experiment_dir=./exper_beta01   --beta=0.1   --dataset=AITEX         --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/AITEX_anomaly_detection       --classname=AITEX           --know_class=None --nAnomaly=1 &&
python train.py --experiment_dir=./exper_beta01   --beta=0.1   --dataset=AITEX         --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/AITEX_anomaly_detection       --classname=AITEX           --know_class=None --nAnomaly=1 &&

python train.py --experiment_dir=./exper_beta01   --beta=0.1   --dataset=elpv          --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/elpv_anomaly_detection        --classname=elpv            --know_class=None --nAnomaly=1 &&
python train.py --experiment_dir=./exper_beta01   --beta=0.1   --dataset=elpv          --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/elpv_anomaly_detection        --classname=elpv            --know_class=None --nAnomaly=1 &&
python train.py --experiment_dir=./exper_beta01   --beta=0.1   --dataset=elpv          --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/elpv_anomaly_detection        --classname=elpv            --know_class=None --nAnomaly=1 &&

python train.py --experiment_dir=./exper_beta01   --beta=0.1   --dataset=hyperkvasir   --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/hyperkvasir_anomaly_detection --classname=hyperkvasir     --know_class=None --nAnomaly=1      --outlier_root=/root/autodl-tmp/ad_wxf/ad_dataset/HeadCT_anomaly_detection &&
python train.py --experiment_dir=./exper_beta01   --beta=0.1   --dataset=hyperkvasir   --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/hyperkvasir_anomaly_detection --classname=hyperkvasir     --know_class=None --nAnomaly=1      --outlier_root=/root/autodl-tmp/ad_wxf/ad_dataset/HeadCT_anomaly_detection &&
python train.py --experiment_dir=./exper_beta01   --beta=0.1   --dataset=hyperkvasir   --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/hyperkvasir_anomaly_detection --classname=hyperkvasir     --know_class=None --nAnomaly=1      --outlier_root=/root/autodl-tmp/ad_wxf/ad_dataset/HeadCT_anomaly_detection &&


python train.py --experiment_dir=./exper_beta05   --beta=0.5  --dataset=mvtecad       --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/mvtec_anomaly_detection       --classname=carpet          --know_class=None --nAnomaly=1 &&
python train.py --experiment_dir=./exper_beta05   --beta=0.5  --dataset=mvtecad       --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/mvtec_anomaly_detection       --classname=carpet          --know_class=None --nAnomaly=1 &&
python train.py --experiment_dir=./exper_beta05   --beta=0.5  --dataset=mvtecad       --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/mvtec_anomaly_detection       --classname=carpet          --know_class=None --nAnomaly=1 &&

python train.py --experiment_dir=./exper_beta05   --beta=0.5   --dataset=AITEX         --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/AITEX_anomaly_detection       --classname=AITEX           --know_class=None --nAnomaly=1 &&
python train.py --experiment_dir=./exper_beta05   --beta=0.5   --dataset=AITEX         --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/AITEX_anomaly_detection       --classname=AITEX           --know_class=None --nAnomaly=1 &&
python train.py --experiment_dir=./exper_beta05   --beta=0.5   --dataset=AITEX         --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/AITEX_anomaly_detection       --classname=AITEX           --know_class=None --nAnomaly=1 &&

python train.py --experiment_dir=./exper_beta05   --beta=0.5   --dataset=elpv          --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/elpv_anomaly_detection        --classname=elpv            --know_class=None --nAnomaly=1 &&
python train.py --experiment_dir=./exper_beta05   --beta=0.5   --dataset=elpv          --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/elpv_anomaly_detection        --classname=elpv            --know_class=None --nAnomaly=1 &&
python train.py --experiment_dir=./exper_beta05   --beta=0.5   --dataset=elpv          --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/elpv_anomaly_detection        --classname=elpv            --know_class=None --nAnomaly=1 &&

python train.py --experiment_dir=./exper_beta05   --beta=0.5   --dataset=hyperkvasir   --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/hyperkvasir_anomaly_detection --classname=hyperkvasir     --know_class=None --nAnomaly=1      --outlier_root=/root/autodl-tmp/ad_wxf/ad_dataset/HeadCT_anomaly_detection &&
python train.py --experiment_dir=./exper_beta05   --beta=0.5   --dataset=hyperkvasir   --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/hyperkvasir_anomaly_detection --classname=hyperkvasir     --know_class=None --nAnomaly=1      --outlier_root=/root/autodl-tmp/ad_wxf/ad_dataset/HeadCT_anomaly_detection &&
python train.py --experiment_dir=./exper_beta05   --beta=0.5   --dataset=hyperkvasir   --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/hyperkvasir_anomaly_detection --classname=hyperkvasir     --know_class=None --nAnomaly=1      --outlier_root=/root/autodl-tmp/ad_wxf/ad_dataset/HeadCT_anomaly_detection 

