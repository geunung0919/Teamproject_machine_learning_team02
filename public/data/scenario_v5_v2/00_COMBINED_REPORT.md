# Scenario v5_v3 Tuned HistGB Generation

## 1. Executive Summary

Generated `scenario_v5_v3` using the selected tuned R3 HistGB scenario predictions. The existing `scenario_v5_v2` folder was preserved.

## 2. Input Data Discovery

| input_path | exists | size_bytes | purpose | used_by_step | note |
| --- | --- | --- | --- | --- | --- |
| data\v5_r3_r6_rf_hist_tuning_v1\scenario_best_model_2026_2030\best_tuned_school_predictions_2026_2030_with_excluded_correction.csv | True | 3780364 | tuned school scenario predictions | scenario_v5_v3_web_export |  |
| data\v5_r3_r6_rf_hist_tuning_v1\scenario_best_model_2026_2030\best_tuned_total_students_by_year_with_excluded_correction.csv | True | 1634 | tuned national totals with excluded correction | scenario_v5_v3_web_export |  |
| public\data\scenario_v5_v2\final_scenario_school_web.csv | True | 9960434 | v2 school web schema and metadata | scenario_v5_v3_web_export |  |
| public\data\scenario_v5_v2\excluded_school_web.csv | True | 1363984 | v2 excluded layer schema and metadata | scenario_v5_v3_web_export |  |
| public\data\scenario_v5_v2\final_scenario_school_year_long.csv | True | 19147530 | v2 long schema | scenario_v5_v3_web_export |  |

## 3. Model Configuration

| item | value |
| --- | --- |
| model_stage | R3 |
| model_family | HistGradientBoostingRegressor |
| param_set | hgb_05_deeper_regularized |
| model_name | R3_multioutput_1to5_incremental_delta_HistGradientBoostingRegressor_hgb_05_deeper_regularized |
| max_iter | 120 |
| max_leaf_nodes | 31 |
| learning_rate | 0.04 |
| l2_regularization | 0.1 |

## 4. Feature Set Used

R3 tuned prediction outputs from `data\v5_r3_r6_rf_hist_tuning_v1\scenario_best_model_2026_2030` were used. No full retraining was executed in this generation task.

## 5. Scenario Generation Method

The script merged tuned P1 school predictions into the v2-compatible school web schema, regenerated 2026-2030 deltas, risk flags, priority ranks, long-form rows, summaries, top tables, and validation reports.

## 6. Output Files Created

| file_path | size_bytes | sha256 | purpose |
| --- | --- | --- | --- |
| public\data\scenario_v5_v3\coordinate_audit.csv | 194 | 87ea42be9827a32b22fe00060f5ea83670069232cd4057b4e6db9d23f0691075 | public web export |
| public\data\scenario_v5_v3\data_dictionary_ko.csv | 11312 | d872db96d60973b9eee839e015c982c630819c07bc642b82e33ee37b27474309 | public web export |
| public\data\scenario_v5_v3\data_quality_audit.csv | 232 | 9a71c75ec9e092d27a726f0fab717d6b2f11bdf4e15c85bb1e674f0c6f4927cf | public web export |
| public\data\scenario_v5_v3\excluded_school_audit.csv | 126 | e006e29ec5daa4492b5a782fd1835ac30a097ed56a981d70446b6e0eeb429c53 | public web export |
| public\data\scenario_v5_v3\excluded_school_data_dictionary_ko.csv | 4019 | fac394af413bad200f6a676afdfd5eeb6192f61ddb6963ce11c73102a934bff7 | public web export |
| public\data\scenario_v5_v3\excluded_school_summary.csv | 1190 | ee26e5925f9b865359085cb672041fc33cd0629943b4ee39130d7b1b0f5279b5 | public web export |
| public\data\scenario_v5_v3\excluded_school_web.csv | 1363984 | 1b730cae911041a7808f44003379a2f2e33419ffbdb5523c3e916ac0cf99800e | public web export |
| public\data\scenario_v5_v3\excluded_school_web.json | 3157236 | f5fe9e5829c2ceaaf14021989f42bec3a7c20d1767712ab99bb1dca66c6a3a41 | public web export |
| public\data\scenario_v5_v3\excluded_school_year_long.csv | 543012 | acddf1f913e56b79747e078aa8b940d2e937ff1a5a7e770784bc2debe96bc39b | public web export |
| public\data\scenario_v5_v3\excluded_school_year_long.json | 1459896 | 32986b5115628a2d325cef05cb60d80cf2d6fbd3a749fc40fff92f72a24bda5a | public web export |
| public\data\scenario_v5_v3\final_scenario_school_web.csv | 9086828 | 9f911112eeb07bbdd7f1598afbcf8b0b1066f51146c3d60387e131ea3b1e7f97 | public web export |
| public\data\scenario_v5_v3\final_scenario_school_web.json | 23010688 | 3e4a352319fa0f3bf704fa3a29a5f3fedbca85d7e3b57797ccc9158a9385f486 | public web export |
| public\data\scenario_v5_v3\final_scenario_school_year_long.csv | 21253456 | 60caa415f32cf65346318efd2c58d42901256ea03f1b9421e1323f25cb3acd84 | public web export |
| public\data\scenario_v5_v3\final_scenario_school_year_long.json | 47032863 | 61e8b756b7bd67be1e1a75c1fa77393f95b788daed28da0e444fd45c422b286b | public web export |
| public\data\scenario_v5_v3\scenario_metadata.json | 1152 | 209d8f4aac9df0ff49c6af63de16b3d87a9fa8e307316db5cc94cd06ba4320f0 | public web export |
| public\data\scenario_v5_v3\scenario_total_validation.csv | 2114 | 94c7b6aa013e80f8650d8176f2c599034a8d5c1e4507beb9781c4dbf2609b0b3 | public web export |
| public\data\scenario_v5_v3\summary_national_by_year.csv | 1095 | f2d96003ccc9eeee422daa9691db24f9b4104a9917d84dde6ba1f4a297412512 | public web export |
| public\data\scenario_v5_v3\summary_school_level_by_year.csv | 2877 | 4957109d0229947a843be7e49c6c16194250f0013abdb07c6699168f4c3da9f0 | public web export |
| public\data\scenario_v5_v3\summary_sgg_by_year.csv | 194230 | cbfa3648aaf8bf912e65b543a23184e7e925d9a0b90c7b232d709dbea8f9896b | public web export |
| public\data\scenario_v5_v3\summary_sido_by_year.csv | 14075 | 5b50a79de19662f9312713dd53e8294d2def80a495fc815b01a7c8b063484541 | public web export |
| public\data\scenario_v5_v3\summary_sido_school_level_by_year.csv | 44683 | 22be4f2d40b21bac5d59ed6c4728a58303e95b943a175fa2ef777f2cdac49c10 | public web export |
| public\data\scenario_v5_v3\top_decline_national_2030.csv | 38071 | 7c9cb3e48af849289f450084dd0ecff3e2dda2b188c92d2af749ed6ba55a4ff2 | public web export |
| public\data\scenario_v5_v3\top_isolated_small_school_2030.csv | 32429 | ef2773650656c48a6025a552ca98ec1cdf408625797e41540622eb4bcc690985 | public web export |
| public\data\scenario_v5_v3\top_priority_by_sido_2030.csv | 118674 | b7be44d1e801822304eb88a1d3dedb1b6493f7a41104b19983305fc3b512da62 | public web export |
| public\data\scenario_v5_v3\top_priority_national_2030.csv | 33131 | e6cf925f0069cb6293053b9ff7dbe90a5dbe1521789bd1cc918a58fa085768b1 | public web export |
| public\data\scenario_v5_v3\web_data_coverage_audit.csv | 381 | 98de8b6f1a9b73496d2af06c81be2380ff2941bedb154d8a1e00510bbd6fdeaf | public web export |

## 7. v2 vs v3 Schema Comparison

| file_name | exists_in_v2 | exists_in_v3 | same_schema | row_count_v2 | row_count_v3 | note |
| --- | --- | --- | --- | --- | --- | --- |
| 00_COMBINED_REPORT.md | True | False | False |  |  | Missing in one version |
| 01_KEY_TABLES.xlsx | True | False | False |  |  | Missing in one version |
| coordinate_audit.csv | True | True | False | 10454 | 2 | CSV columns differ |
| data_dictionary_ko.csv | True | True | True | 76 | 76 |  |
| data_quality_audit.csv | True | True | False | 10 | 6 | CSV columns differ |
| excluded_school_audit.csv | True | True | False | 8 | 3 | CSV columns differ |
| excluded_school_data_dictionary_ko.csv | True | True | True | 30 | 30 |  |
| excluded_school_summary.csv | True | True | False | 162 | 50 | CSV columns differ |
| excluded_school_web.csv | True | True | True | 2173 | 2173 |  |
| excluded_school_web.json | True | True | True | 2173 | 2173 |  |
| excluded_school_year_long.csv | True | True | True | 2173 | 2173 |  |
| excluded_school_year_long.json | True | True | True | 2173 | 2173 |  |
| final_scenario_school_web.csv | True | True | True | 10454 | 10454 |  |
| final_scenario_school_web.json | True | True | True | 10454 | 10454 |  |
| final_scenario_school_year_long.csv | True | True | True | 62724 | 62724 |  |
| final_scenario_school_year_long.json | True | True | True | 62724 | 62724 |  |
| scenario_metadata.json | True | True | False | 13 | 16 | JSON keys differ |
| scenario_total_validation.csv | True | True | False | 1 | 6 | CSV columns differ |
| summary_national_by_year.csv | True | True | True | 6 | 6 |  |
| summary_school_level_by_year.csv | True | True | True | 18 | 18 |  |
| summary_sgg_by_year.csv | True | True | True | 1374 | 1374 |  |
| summary_sido_by_year.csv | True | True | True | 102 | 102 |  |
| summary_sido_school_level_by_year.csv | True | True | True | 306 | 306 |  |
| top_decline_national_2030.csv | True | True | True | 100 | 100 |  |
| top_isolated_small_school_2030.csv | True | True | True | 100 | 100 |  |
| top_priority_by_sido_2030.csv | True | True | True | 340 | 340 |  |
| top_priority_national_2030.csv | True | True | True | 100 | 100 |  |
| web_data_coverage_audit.csv | True | True | False | 13 | 4 | CSV columns differ |

## 8. Validation Results

Validation result: **PASS**

| item | value |
| --- | --- |
| validation_result | PASS |
| scenario_version | scenario_v5_v3 |
| model_name | R3_multioutput_1to5_incremental_delta_HistGradientBoostingRegressor_hgb_05_deeper_regularized |
| public_output | public\data\scenario_v5_v3 |
| data_output | data\v5_tuned_histgb_scenario_v3 |

## 9. Scenario Sanity Checks

| check | status | value |
| --- | --- | --- |
| scenario_years_cover_2026_2030 | PASS | 2025,2026,2027,2028,2029,2030 |
| required_public_files_exist | PASS |  |
| no_negative_predictions | PASS | 0 |
| school_count_matches_v2 | PASS | 10454 vs 10454 |
| national_total_2030_p1 | PASS | 3257211.8419410978 |
| corrected_total_2030_with_excluded | PASS | 4213774.284467546 |
| model_metadata | PASS | R3_multioutput_1to5_incremental_delta_HistGradientBoostingRegressor_hgb_05_deeper_regularized |

## 10. Web Compatibility Check

| check | status | value |
| --- | --- | --- |
| v2_folder_preserved | PASS | True |
| final_json_schema_matches_v2 | PASS |  |
| excluded_json_schema_matches_v2 | PASS |  |
| summary_national_schema_matches_v2 | PASS |  |
| web_source_modified | PASS | NO |

## 11. Known Limitations

The excluded/event layer is preserved as a separate web layer. Corrected national totals including excluded/event schools are available in `scenario_total_validation.csv`.

## 12. Next Steps

After manual review, point the web app data path from `scenario_v5_v2` to `scenario_v5_v3`.

## 13. Final Judgment

PASS


