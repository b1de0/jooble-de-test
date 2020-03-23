docker-compose up --build
{ head -n1 output/tmp/0.part; for f in output/tmp/*.part; do tail -n+2 "$f"; done; } > output/test_proc.tsv
rm -r output/tmp