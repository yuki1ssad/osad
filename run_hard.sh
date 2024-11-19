# python train.py --dataset=mvtecad --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/mvtec_anomaly_detection --classname=carpet --know_class=color --nAnomaly=1 &&
# python train.py --dataset=mvtecad --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/mvtec_anomaly_detection --classname=carpet --know_class=cut --nAnomaly=1 &&
# python train.py --dataset=mvtecad --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/mvtec_anomaly_detection --classname=carpet --know_class=hole --nAnomaly=1 &&
# python train.py --dataset=mvtecad --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/mvtec_anomaly_detection --classname=carpet --know_class=metal --nAnomaly=1 &&
# python train.py --dataset=mvtecad --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/mvtec_anomaly_detection --classname=carpet --know_class=thread --nAnomaly=1 

python train.py --dataset=mvtecad       --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/mvtec_anomaly_detection       --classname=metal_nut       --know_class=bent                   --nAnomaly=1 &&
python train.py --dataset=mvtecad       --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/mvtec_anomaly_detection       --classname=metal_nut       --know_class=color                  --nAnomaly=1 &&
python train.py --dataset=mvtecad       --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/mvtec_anomaly_detection       --classname=metal_nut       --know_class=flip                   --nAnomaly=1 &&
python train.py --dataset=mvtecad       --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/mvtec_anomaly_detection       --classname=metal_nut       --know_class=scratch                --nAnomaly=1 &&

python train.py --dataset=AITEX         --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/AITEX_anomaly_detection       --classname=AITEX           --know_class=Broken_end             --nAnomaly=1 &&
python train.py --dataset=AITEX         --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/AITEX_anomaly_detection       --classname=AITEX           --know_class=Broken_pick            --nAnomaly=1 &&
python train.py --dataset=AITEX         --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/AITEX_anomaly_detection       --classname=AITEX           --know_class=Cut_selvage            --nAnomaly=1 &&
python train.py --dataset=AITEX         --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/AITEX_anomaly_detection       --classname=AITEX           --know_class=Fuzzyball              --nAnomaly=1 &&
python train.py --dataset=AITEX         --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/AITEX_anomaly_detection       --classname=AITEX           --know_class=Nep                    --nAnomaly=1 &&
python train.py --dataset=AITEX         --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/AITEX_anomaly_detection       --classname=AITEX           --know_class=Weft_crack             --nAnomaly=1 &&

python train.py --dataset=elpv          --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/elpv_anomaly_detection        --classname=elpv            --know_class=mono                   --nAnomaly=1 &&
python train.py --dataset=elpv          --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/elpv_anomaly_detection        --classname=elpv            --know_class=poly                   --nAnomaly=1 &&

python train.py --dataset=mastcam       --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/MastCam_anomaly_detection     --classname=mastcam         --know_class=bedrock                --nAnomaly=1 &&
python train.py --dataset=mastcam       --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/MastCam_anomaly_detection     --classname=mastcam         --know_class=broken-rock            --nAnomaly=1 &&
python train.py --dataset=mastcam       --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/MastCam_anomaly_detection     --classname=mastcam         --know_class=drill-hole             --nAnomaly=1 &&
python train.py --dataset=mastcam       --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/MastCam_anomaly_detection     --classname=mastcam         --know_class=drt                    --nAnomaly=1 &&
python train.py --dataset=mastcam       --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/MastCam_anomaly_detection     --classname=mastcam         --know_class=dump-pile              --nAnomaly=1 &&
python train.py --dataset=mastcam       --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/MastCam_anomaly_detection     --classname=mastcam         --know_class=float                  --nAnomaly=1 &&
python train.py --dataset=mastcam       --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/MastCam_anomaly_detection     --classname=mastcam         --know_class=meteorite              --nAnomaly=1 &&
python train.py --dataset=mastcam       --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/MastCam_anomaly_detection     --classname=mastcam         --know_class=scuff                  --nAnomaly=1 &&
python train.py --dataset=mastcam       --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/MastCam_anomaly_detection     --classname=mastcam         --know_class=veins                  --nAnomaly=1 &&

python train.py --dataset=hyperkvasir   --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/hyperkvasir_anomaly_detection --classname=hyperkvasir     --know_class=barretts               --nAnomaly=1 &&
python train.py --dataset=hyperkvasir   --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/hyperkvasir_anomaly_detection --classname=hyperkvasir     --know_class=barretts-short-segment --nAnomaly=1 &&
python train.py --dataset=hyperkvasir   --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/hyperkvasir_anomaly_detection --classname=hyperkvasir     --know_class=esophagitis-a          --nAnomaly=1 &&
python train.py --dataset=hyperkvasir   --dataset_root=/root/autodl-tmp/ad_wxf/ad_dataset/hyperkvasir_anomaly_detection --classname=hyperkvasir     --know_class=esophagitis-b-d        --nAnomaly=1


# nohup ./run_hard.sh > hard_mvtec_1.log 2>&1 &