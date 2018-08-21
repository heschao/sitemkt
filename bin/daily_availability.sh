
docker-compose -f docker/docker-compose-prod.yml run sitemkt-console \
  python sitemkt/dates_available.py --park-id 70030 --max-pages 2