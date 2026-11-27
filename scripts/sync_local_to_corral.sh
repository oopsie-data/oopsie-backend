LOCAL_DIR=/scratch/11168/cvoelcker/oopsie-scratch

rsync -rchv --exclude 'OopsieData-Submissions/' --exclude 'annotation/'  $LOCAL_DIR/oopsiedata /corral/utexas/IRI26024/
rsync -rhv --size-only  $LOCAL_DIR/oopsiedata /corral/utexas/IRI26024/
