# nohup ./ablation_numProtos.sh > ablation_numProtos.log 2>&1 &


python train.py --experiment_dir=./exper_np2   --numProtos=2  --dataset=mvtecad       --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/mvtec_anomaly_detection       --classname=carpet          --know_class=None --nAnomaly=1 &&
python train.py --experiment_dir=./exper_np2   --numProtos=2  --dataset=mvtecad       --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/mvtec_anomaly_detection       --classname=carpet          --know_class=None --nAnomaly=1 &&
python train.py --experiment_dir=./exper_np2   --numProtos=2  --dataset=mvtecad       --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/mvtec_anomaly_detection       --classname=carpet          --know_class=None --nAnomaly=1 &&

python train.py --experiment_dir=./exper_np2   --numProtos=2   --dataset=AITEX         --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/AITEX_anomaly_detection       --classname=AITEX           --know_class=None --nAnomaly=1 &&
python train.py --experiment_dir=./exper_np2   --numProtos=2   --dataset=AITEX         --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/AITEX_anomaly_detection       --classname=AITEX           --know_class=None --nAnomaly=1 &&
python train.py --experiment_dir=./exper_np2   --numProtos=2   --dataset=AITEX         --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/AITEX_anomaly_detection       --classname=AITEX           --know_class=None --nAnomaly=1 &&

python train.py --experiment_dir=./exper_np2   --numProtos=2   --dataset=elpv          --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/elpv_anomaly_detection        --classname=elpv            --know_class=None --nAnomaly=1 &&
python train.py --experiment_dir=./exper_np2   --numProtos=2   --dataset=elpv          --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/elpv_anomaly_detection        --classname=elpv            --know_class=None --nAnomaly=1 &&
python train.py --experiment_dir=./exper_np2   --numProtos=2   --dataset=elpv          --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/elpv_anomaly_detection        --classname=elpv            --know_class=None --nAnomaly=1 &&

python train.py --experiment_dir=./exper_np2   --numProtos=2   --dataset=hyperkvasir   --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/hyperkvasir_anomaly_detection --classname=hyperkvasir     --know_class=None --nAnomaly=1      --outlier_root=/root/autodl-tmp/ad_wxf/ad_dataset/HeadCT_anomaly_detection &&
python train.py --experiment_dir=./exper_np2   --numProtos=2   --dataset=hyperkvasir   --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/hyperkvasir_anomaly_detection --classname=hyperkvasir     --know_class=None --nAnomaly=1      --outlier_root=/root/autodl-tmp/ad_wxf/ad_dataset/HeadCT_anomaly_detection &&
python train.py --experiment_dir=./exper_np2   --numProtos=2   --dataset=hyperkvasir   --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/hyperkvasir_anomaly_detection --classname=hyperkvasir     --know_class=None --nAnomaly=1      --outlier_root=/root/autodl-tmp/ad_wxf/ad_dataset/HeadCT_anomaly_detection &&

#------------------------------------------------------------------------------------------------------------#

python train.py --experiment_dir=./exper_np4   --numProtos=4  --dataset=mvtecad       --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/mvtec_anomaly_detection       --classname=carpet          --know_class=None --nAnomaly=1 &&
python train.py --experiment_dir=./exper_np4   --numProtos=4  --dataset=mvtecad       --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/mvtec_anomaly_detection       --classname=carpet          --know_class=None --nAnomaly=1 &&
python train.py --experiment_dir=./exper_np4   --numProtos=4  --dataset=mvtecad       --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/mvtec_anomaly_detection       --classname=carpet          --know_class=None --nAnomaly=1 &&

python train.py --experiment_dir=./exper_np4   --numProtos=4   --dataset=AITEX         --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/AITEX_anomaly_detection       --classname=AITEX           --know_class=None --nAnomaly=1 &&
python train.py --experiment_dir=./exper_np4   --numProtos=4   --dataset=AITEX         --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/AITEX_anomaly_detection       --classname=AITEX           --know_class=None --nAnomaly=1 &&
python train.py --experiment_dir=./exper_np4   --numProtos=4   --dataset=AITEX         --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/AITEX_anomaly_detection       --classname=AITEX           --know_class=None --nAnomaly=1 &&

python train.py --experiment_dir=./exper_np4   --numProtos=4   --dataset=elpv          --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/elpv_anomaly_detection        --classname=elpv            --know_class=None --nAnomaly=1 &&
python train.py --experiment_dir=./exper_np4   --numProtos=4   --dataset=elpv          --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/elpv_anomaly_detection        --classname=elpv            --know_class=None --nAnomaly=1 &&
python train.py --experiment_dir=./exper_np4   --numProtos=4   --dataset=elpv          --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/elpv_anomaly_detection        --classname=elpv            --know_class=None --nAnomaly=1 &&

python train.py --experiment_dir=./exper_np4   --numProtos=4   --dataset=hyperkvasir   --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/hyperkvasir_anomaly_detection --classname=hyperkvasir     --know_class=None --nAnomaly=1      --outlier_root=/root/autodl-tmp/ad_wxf/ad_dataset/HeadCT_anomaly_detection &&
python train.py --experiment_dir=./exper_np4   --numProtos=4   --dataset=hyperkvasir   --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/hyperkvasir_anomaly_detection --classname=hyperkvasir     --know_class=None --nAnomaly=1      --outlier_root=/root/autodl-tmp/ad_wxf/ad_dataset/HeadCT_anomaly_detection &&
python train.py --experiment_dir=./exper_np4   --numProtos=4   --dataset=hyperkvasir   --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/hyperkvasir_anomaly_detection --classname=hyperkvasir     --know_class=None --nAnomaly=1      --outlier_root=/root/autodl-tmp/ad_wxf/ad_dataset/HeadCT_anomaly_detection &&

#------------------------------------------------------------------------------------------------------------#

python train.py --experiment_dir=./exper_np5   --numProtos=5  --dataset=mvtecad       --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/mvtec_anomaly_detection       --classname=carpet          --know_class=None --nAnomaly=1 &&
python train.py --experiment_dir=./exper_np5   --numProtos=5  --dataset=mvtecad       --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/mvtec_anomaly_detection       --classname=carpet          --know_class=None --nAnomaly=1 &&
python train.py --experiment_dir=./exper_np5   --numProtos=5  --dataset=mvtecad       --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/mvtec_anomaly_detection       --classname=carpet          --know_class=None --nAnomaly=1 &&

python train.py --experiment_dir=./exper_np5   --numProtos=5   --dataset=AITEX         --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/AITEX_anomaly_detection       --classname=AITEX           --know_class=None --nAnomaly=1 &&
python train.py --experiment_dir=./exper_np5   --numProtos=5   --dataset=AITEX         --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/AITEX_anomaly_detection       --classname=AITEX           --know_class=None --nAnomaly=1 &&
python train.py --experiment_dir=./exper_np5   --numProtos=5   --dataset=AITEX         --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/AITEX_anomaly_detection       --classname=AITEX           --know_class=None --nAnomaly=1 &&

python train.py --experiment_dir=./exper_np5   --numProtos=5   --dataset=elpv          --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/elpv_anomaly_detection        --classname=elpv            --know_class=None --nAnomaly=1 &&
python train.py --experiment_dir=./exper_np5   --numProtos=5   --dataset=elpv          --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/elpv_anomaly_detection        --classname=elpv            --know_class=None --nAnomaly=1 &&
python train.py --experiment_dir=./exper_np5   --numProtos=5   --dataset=elpv          --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/elpv_anomaly_detection        --classname=elpv            --know_class=None --nAnomaly=1 &&

python train.py --experiment_dir=./exper_np5   --numProtos=5   --dataset=hyperkvasir   --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/hyperkvasir_anomaly_detection --classname=hyperkvasir     --know_class=None --nAnomaly=1      --outlier_root=/root/autodl-tmp/ad_wxf/ad_dataset/HeadCT_anomaly_detection &&
python train.py --experiment_dir=./exper_np5   --numProtos=5   --dataset=hyperkvasir   --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/hyperkvasir_anomaly_detection --classname=hyperkvasir     --know_class=None --nAnomaly=1      --outlier_root=/root/autodl-tmp/ad_wxf/ad_dataset/HeadCT_anomaly_detection &&
python train.py --experiment_dir=./exper_np5   --numProtos=5   --dataset=hyperkvasir   --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/hyperkvasir_anomaly_detection --classname=hyperkvasir     --know_class=None --nAnomaly=1      --outlier_root=/root/autodl-tmp/ad_wxf/ad_dataset/HeadCT_anomaly_detection 



