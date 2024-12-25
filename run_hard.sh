python train.py --dataset=AITEX         --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/AITEX_anomaly_detection       --classname=AITEX           --know_class=Broken_pick            --nAnomaly=1 &&
python train.py --dataset=AITEX         --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/AITEX_anomaly_detection       --classname=AITEX           --know_class=Broken_pick            --nAnomaly=1 &&
python train.py --dataset=AITEX         --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/AITEX_anomaly_detection       --classname=AITEX           --know_class=Broken_pick            --nAnomaly=1 &&

python train.py --dataset=mastcam       --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/MastCam_anomaly_detection     --classname=mastcam         --know_class=bedrock                --nAnomaly=1 &&
python train.py --dataset=mastcam       --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/MastCam_anomaly_detection     --classname=mastcam         --know_class=bedrock                --nAnomaly=1 &&
python train.py --dataset=mastcam       --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/MastCam_anomaly_detection     --classname=mastcam         --know_class=bedrock                --nAnomaly=1 &&

python train.py --dataset=mastcam       --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/MastCam_anomaly_detection     --classname=mastcam         --know_class=broken-rock            --nAnomaly=1 &&
python train.py --dataset=mastcam       --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/MastCam_anomaly_detection     --classname=mastcam         --know_class=broken-rock            --nAnomaly=1 &&
python train.py --dataset=mastcam       --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/MastCam_anomaly_detection     --classname=mastcam         --know_class=broken-rock            --nAnomaly=1 &&

python train.py --dataset=mastcam       --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/MastCam_anomaly_detection     --classname=mastcam         --know_class=dump-pile              --nAnomaly=1 &&
python train.py --dataset=mastcam       --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/MastCam_anomaly_detection     --classname=mastcam         --know_class=dump-pile              --nAnomaly=1 &&
python train.py --dataset=mastcam       --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/MastCam_anomaly_detection     --classname=mastcam         --know_class=dump-pile              --nAnomaly=1 &&

python train.py --dataset=mastcam       --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/MastCam_anomaly_detection     --classname=mastcam         --know_class=float                  --nAnomaly=1 &&
python train.py --dataset=mastcam       --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/MastCam_anomaly_detection     --classname=mastcam         --know_class=float                  --nAnomaly=1 &&
python train.py --dataset=mastcam       --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/MastCam_anomaly_detection     --classname=mastcam         --know_class=float                  --nAnomaly=1 &&

python train.py --dataset=mastcam       --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/MastCam_anomaly_detection     --classname=mastcam         --know_class=veins                  --nAnomaly=1 &&
python train.py --dataset=mastcam       --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/MastCam_anomaly_detection     --classname=mastcam         --know_class=veins                  --nAnomaly=1 &&
python train.py --dataset=mastcam       --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/MastCam_anomaly_detection     --classname=mastcam         --know_class=veins                  --nAnomaly=1 &&

python train.py --dataset=hyperkvasir   --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/hyperkvasir_anomaly_detection --classname=hyperkvasir     --know_class=barretts               --nAnomaly=1    --outlier_root=/root/autodl-tmp/ad_wxf/ad_dataset/HeadCT_anomaly_detection &&
python train.py --dataset=hyperkvasir   --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/hyperkvasir_anomaly_detection --classname=hyperkvasir     --know_class=barretts               --nAnomaly=1    --outlier_root=/root/autodl-tmp/ad_wxf/ad_dataset/HeadCT_anomaly_detection &&
python train.py --dataset=hyperkvasir   --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/hyperkvasir_anomaly_detection --classname=hyperkvasir     --know_class=barretts               --nAnomaly=1    --outlier_root=/root/autodl-tmp/ad_wxf/ad_dataset/HeadCT_anomaly_detection &&




#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

python train.py --dataset=mastcam       --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/MastCam_anomaly_detection     --classname=mastcam         --know_class=bedrock                --nAnomaly=10 &&
python train.py --dataset=mastcam       --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/MastCam_anomaly_detection     --classname=mastcam         --know_class=bedrock                --nAnomaly=10 &&
python train.py --dataset=mastcam       --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/MastCam_anomaly_detection     --classname=mastcam         --know_class=bedrock                --nAnomaly=10 &&
python train.py --dataset=mastcam       --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/MastCam_anomaly_detection     --classname=mastcam         --know_class=bedrock                --nAnomaly=10 &&
python train.py --dataset=mastcam       --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/MastCam_anomaly_detection     --classname=mastcam         --know_class=bedrock                --nAnomaly=10 

# nohup ./run_hard.sh > hard.log 2>&1 &