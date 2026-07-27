# LUBM split=test ged_column=3 checked_pairs=10000
| method | pairs | mae | acc | fea |
| --- | --- | --- | --- | --- |
| size_delta | 10000 | 4.5729 | 0.0152 | 0.0152 |
| random_unit | 10000 | 4.9551 | 0.0041 | 1.0 |
| entity_id_unit | 10000 | 3.6328 | 0.0323 | 1.0 |
| feature_greedy_unit | 10000 | 4.1557 | 0.0227 | 1.0 |

# SWDF split=test ged_column=3 checked_pairs=10000
| method | pairs | mae | acc | fea |
| --- | --- | --- | --- | --- |
| size_delta | 10000 | 4.7799 | 0.0317 | 0.061 |
| random_unit | 10000 | 4.2563 | 0.0052 | 1.0 |
| entity_id_unit | 10000 | 2.9668 | 0.0468 | 1.0 |
| feature_greedy_unit | 10000 | 3.6474 | 0.0259 | 1.0 |

# YAGO split=test ged_column=3 checked_pairs=6000
| method | pairs | mae | acc | fea |
| --- | --- | --- | --- | --- |
| size_delta | 6000 | 0.0 | 1.0 | 1.0 |
| random_unit | 6000 | 39.6858 | 0.0 | 1.0 |
| entity_id_unit | 6000 | 39.3618 | 0.0 | 1.0 |
| feature_greedy_unit | 6000 | 39.3618 | 0.0 | 1.0 |

# WIKIDATA split=test ged_column=3 checked_pairs=10000
| method | pairs | mae | acc | fea |
| --- | --- | --- | --- | --- |
| size_delta | 10000 | 0.0 | 1.0 | 1.0 |
| random_unit | 10000 | 53.1767 | 0.0003 | 1.0 |
| entity_id_unit | 10000 | 52.8581 | 0.0001 | 1.0 |
| feature_greedy_unit | 10000 | 52.8581 | 0.0001 | 1.0 |
