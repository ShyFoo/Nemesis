# original
CUDA_VISIBLE_DEVICES=0 bash scripts/coop_crt/eval.sh rn50_ep50 end 16 1 False 50 original 666 666
CUDA_VISIBLE_DEVICES=0 bash scripts/coop_crt/eval.sh rn50_ep50 end 16 2 False 50 original 666 666
CUDA_VISIBLE_DEVICES=0 bash scripts/coop_crt/eval.sh rn50_ep100 end 16 4 False 100 original 666 666
CUDA_VISIBLE_DEVICES=0 bash scripts/coop_crt/eval.sh rn50_ep100 end 16 8 False 100 original 666 666
CUDA_VISIBLE_DEVICES=0 bash scripts/coop_crt/eval.sh rn50 end 16 16 False 200 original 666 666
# replace
CUDA_VISIBLE_DEVICES=0 bash scripts/coop_crt/eval.sh rn50_ep50 end 16 1 False 100 replace 0 0.
CUDA_VISIBLE_DEVICES=0 bash scripts/coop_crt/eval.sh rn50_ep50 end 16 1 False 100 replace 1 0.
CUDA_VISIBLE_DEVICES=0 bash scripts/coop_crt/eval.sh rn50_ep50 end 16 1 False 100 replace 2 0.
# rescale
CUDA_VISIBLE_DEVICES=0 bash scripts/coop_crt/eval.sh rn50_ep50 end 16 1 False 100 scale 0 0.001
CUDA_VISIBLE_DEVICES=0 bash scripts/coop_crt/eval.sh rn50_ep50 end 16 1 False 100 scale 1 0.001
CUDA_VISIBLE_DEVICES=0 bash scripts/coop_crt/eval.sh rn50_ep50 end 16 1 False 100 scale 2 0.001
