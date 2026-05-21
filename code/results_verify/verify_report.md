# Verify report

Comparing `/Users/sanjiangli/Documents/pyzoo/dsabre/code/results` (paper) vs `/Users/sanjiangli/Documents/pyzoo/dsabre/code/results_verify` (verify).

Thresholds: noise ≤ 2.0%, material > 10.0%.


## Summary

| file | status |
|---|---|
| `regen_ablation_corners.json` | **match** |
| `results_100q.json` | **material** |
| `results_200q.json` | **material** |
| `results_25q.json` | **match** |
| `results_25q_8links.json` | **match** |
| `results_360q.json` | **material** |
| `results_36q.json` | **match** |
| `results_36q_8links.json` | **match** |
| `results_64q.json` | **match** |
| `results_dmaps_bench.json` | **match** |
| `results_fill_sweep_dmaps.json` | **match** |
| `results_fill_sweep_dse.json` | **match** |
| `results_pytket_large.json` | **match** |
| `results_pytket_layout.json` | **material** |


## regen_ablation_corners.json — **match** (0 keys with diffs)


## results_100q.json — **material** (2 keys with diffs)


### `100q/qft`
| metric | paper | verify | Δ% |
|---|---|---|---|
| `/routers/dSE/eprs` | 301 | 319 | +6.0% |
| `/routers/dSE/ls` | 4077 | 4711 | +15.6% |

### `100q/qpeexact`
| metric | paper | verify | Δ% |
|---|---|---|---|
| `/routers/dSE/eprs` | 217 | 274 | +26.3% |
| `/routers/dSE/ls` | 4139 | 4559 | +10.1% |

## results_200q.json — **material** (2 keys with diffs)


### `200q/qft`
| metric | paper | verify | Δ% |
|---|---|---|---|
| `/routers/dSE/eprs` | 1151 | 1014 | -11.9% |
| `/routers/dSE/ls` | 10989 | 9455 | -14.0% |

### `200q/qpeexact`
| metric | paper | verify | Δ% |
|---|---|---|---|
| `/routers/dSE/eprs` | 1340 | 1158 | -13.6% |
| `/routers/dSE/ls` | 12091 | 10887 | -10.0% |

## results_25q.json — **match** (0 keys with diffs)


## results_25q_8links.json — **match** (0 keys with diffs)


## results_360q.json — **material** (2 keys with diffs)


### `360q/qft`
| metric | paper | verify | Δ% |
|---|---|---|---|
| `/routers/dSE/eprs` | 579 | 670 | +15.7% |
| `/routers/dSE/ls` | 27489 | 29831 | +8.5% |
| `/ts/teledata` | None | 1107 | — |
| `/ts/telegate` | None | 2 | — |

### `360q/qpeexact`
| metric | paper | verify | Δ% |
|---|---|---|---|
| `/routers/dSE/eprs` | 863 | None | — |
| `/routers/dSE/ls` | 26779 | None | — |
| `/ts/eprs` | 1071 | None | — |

## results_36q.json — **match** (0 keys with diffs)


## results_36q_8links.json — **match** (0 keys with diffs)


## results_64q.json — **match** (0 keys with diffs)


## results_dmaps_bench.json — **match** (0 keys with diffs)


## results_fill_sweep_dmaps.json — **match** (0 keys with diffs)


## results_fill_sweep_dse.json — **match** (0 keys with diffs)


## results_pytket_large.json — **match** (0 keys with diffs)


## results_pytket_layout.json — **material** (14 keys with diffs)


### `25q/25q/ae`
| metric | paper | verify | Δ% |
|---|---|---|---|
| `/routers/dSE/eprs` | 56 | 23 | -58.9% |
| `/routers/dSE/ls` | 407 | 366 | -10.1% |

### `25q/25q/qft`
| metric | paper | verify | Δ% |
|---|---|---|---|
| `/routers/dSE/ls` | 437 | 563 | +28.8% |

### `25q/25q/qnn`
| metric | paper | verify | Δ% |
|---|---|---|---|
| `/routers/dSE/ls` | 1179 | 1077 | -8.7% |

### `25q/25q/random`
| metric | paper | verify | Δ% |
|---|---|---|---|
| `/routers/dSE/ls` | 1349 | 1412 | +4.7% |

### `36q/36q/dj`
| metric | paper | verify | Δ% |
|---|---|---|---|
| `/routers/dSE/eprs` | 10 | 7 | -30.0% |
| `/routers/dSE/ls` | 58 | 55 | -5.2% |

### `36q/36q/qaoa`
| metric | paper | verify | Δ% |
|---|---|---|---|
| `/routers/dSE/eprs` | 139 | 156 | +12.2% |
| `/routers/dSE/ls` | 1164 | 1286 | +10.5% |

### `36q/36q/qpeexact`
| metric | paper | verify | Δ% |
|---|---|---|---|
| `/routers/dSE/eprs` | 70 | 87 | +24.3% |
| `/routers/dSE/ls` | 733 | 1103 | +50.5% |

### `36q/36q/vqe_su2`
| metric | paper | verify | Δ% |
|---|---|---|---|
| `/routers/dSE/ls` | 101 | 106 | +5.0% |

### `36q/36q/wstate`
| metric | paper | verify | Δ% |
|---|---|---|---|
| `/routers/dSE/eprs` | 6 | 8 | +33.3% |
| `/routers/dSE/ls` | 46 | 56 | +21.7% |

### `64q/64q/ae`
| metric | paper | verify | Δ% |
|---|---|---|---|
| `/routers/dSE/eprs` | 211 | 226 | +7.1% |
| `/routers/dSE/ls` | 1975 | 2170 | +9.9% |

### `64q/64q/graphstate`
| metric | paper | verify | Δ% |
|---|---|---|---|
| `/routers/dSE/ls` | 54 | 64 | +18.5% |

### `64q/64q/qft`
| metric | paper | verify | Δ% |
|---|---|---|---|
| `/routers/dSE/eprs` | 219 | 229 | +4.6% |
| `/routers/dSE/ls` | 2184 | 2347 | +7.5% |

### `64q/64q/qnn`
| metric | paper | verify | Δ% |
|---|---|---|---|
| `/routers/dSE/eprs` | 597 | 518 | -13.2% |
| `/routers/dSE/ls` | 8385 | 7949 | -5.2% |

### `64q/64q/random`
| metric | paper | verify | Δ% |
|---|---|---|---|
| `/routers/dSE/eprs` | 792 | 757 | -4.4% |
| `/routers/dSE/ls` | 4053 | 3956 | -2.4% |