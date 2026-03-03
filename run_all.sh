#!/usr/bin/env bash

set -e  # stop on error

SUBSETS=(
  BEIJING_SUBWAY_30MIN
  HZMETRO
  LOS_LOOP
  PEMS03
  PEMS04
  PEMS07
  PEMS08
  PEMS_BAY
  Q-TRAFFIC
  SHMETRO
  alibaba_cluster_trace_2018
  australian_electricity_demand
  azure_vm_traces_2017
  bdg-2_bear
  bdg-2_fox
  bdg-2_panther
  bdg-2_rat
  beijing_air_quality
  bitcoin_with_missing
  borealis
  borg_cluster_data_2011
  buildings_900k
  bull
  cdc_fluview_ilinet
  cdc_fluview_who_nrevss
  china_air_quality
  cif_2016_12
  cif_2016_6
  cmip6_1850
  cmip6_1855
  cmip6_1860
  cmip6_1865
  cmip6_1870
  cmip6_1875
  cmip6_1880
  cmip6_1885
  cmip6_1890
  cmip6_1895
  cmip6_1900
  cmip6_1905
  cmip6_1910
  cmip6_1915
  cmip6_1920
  cmip6_1925
  cmip6_1930
  cmip6_1935
  cmip6_1940
  cmip6_1945
  cmip6_1950
  cmip6_1955
  cmip6_1960
  cmip6_1965
  cmip6_1970
  cmip6_1975
  cmip6_1980
  cmip6_1985
  cmip6_1990
  cmip6_1995
  cmip6_2000
  cmip6_2005
  cmip6_2010
  cockatoo
  covid19_energy
  covid_mobility
  default
  elecdemand
  elf
  era5_1989
  era5_1990
  era5_1991
  era5_1992
  era5_1993
  era5_1994
  era5_1995
  era5_1996
  era5_1997
  era5_1998
  era5_1999
  era5_2000
  era5_2001
  era5_2002
  era5_2003
  era5_2004
  era5_2005
  era5_2006
  era5_2007
  era5_2008
  era5_2009
  era5_2010
  era5_2011
  era5_2012
  era5_2013
  era5_2014
  era5_2015
  era5_2016
  era5_2017
  era5_2018
  extended_web_traffic_with_missing
  favorita_sales
  favorita_transactions
  fred_md
  gfc14_load
  godaddy
  hog
  ideal
  kaggle_web_traffic_weekly
  kdd2022
  largest_2017
  largest_2018
  largest_2019
  largest_2020
  largest_2021
  lcl
  london_smart_meters_with_missing
  m1_monthly
  m1_quarterly
  m1_yearly
  m5
  monash_m3_monthly
  monash_m3_other
  monash_m3_quarterly
  monash_m3_yearly
  nn5_daily_with_missing
  nn5_weekly
  oikolab_weather
  pedestrian_counts
  project_tycho
  residential_load_power
  residential_pv_power
  rideshare_with_missing
  sceaux
  smart
  solar_power
  subseasonal
  subseasonal_precip
  sunspot_with_missing
  taxi_30min
  tourism_monthly
  tourism_quarterly
  tourism_yearly
  traffic_hourly
  traffic_weekly
  uber_tlc_daily
  uber_tlc_hourly
  vehicle_trips_with_missing
  weather
  wiki-rolling_nips
  wind_farms_with_missing
  wind_power
)

for subset in "${SUBSETS[@]}"; do
  echo "Running subset: $subset"

  CUDA_VISIBLE_DEVICES=3 uv run python scripts/train_mixture_online.py \
    --subsets "$subset" \
    --model_names chronos2 moirai2 \
    --loss mase \
    --lr 0.01 \
    --epochs 10 \
    --val_every_n_epochs 5 \
    --batch_size 32 \
    --max_series_per_subset 10

  echo "Finished subset: $subset"
  echo "----------------------------------------"
done