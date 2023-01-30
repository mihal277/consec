find -L "$DIRECTORY" -type f -name "*.conllu" | \
xargs -I % \
python src/scripts/model/make_sense_frequency_list.py \
model.model_checkpoint=experiments/released-ckpts/consec_semcor_normal_best.ckpt \
amalgum_path=% model.device=0