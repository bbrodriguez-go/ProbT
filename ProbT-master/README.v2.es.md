# probt v2 — Motor de Probabilidad Direccional Condicionado a Zonas

*Informe final honesto. Espejo del original en inglés [README.v2.md](README.v2.md).
La documentación de la plataforma v1 está en [README.md](README.md).*

## Qué es v2

v1 preguntaba, en cada barra: *"¿se moverá el precio +2·ATR antes que −1·ATR?"* — y
reportaba honestamente que no había edge (Brier ≈ 0.266 vs. 0.25 aleatorio).

v2 hace una pregunta más afilada, y solo en puntos de presión: **cuando el precio toca una
zona de oferta/demanda, un order block o un FVG — ¿la zona aguanta (bounce) o falla
(break)?** Dos modelos separados por (símbolo, timeframe), entrenados solo sobre eventos
de toque de zona, con validación walk-forward purgada, calibración de Platt, intervalos
split-conformal y sizing de Kelly con payoff por evento.

## Resumen ejecutivo (leer antes de confiar en cualquier número)

1. **El modelo de break tiene skill real y calibrado.** XAUUSD 1H fuera de muestra:
   Brier 0.194, **BSS +0.153**, **ROC-AUC 0.740**, ECE 0.040, bien calibrado en regímenes
   VIX calmado/normal/estresado por igual. 4H confirma direccionalmente (BSS +0.116,
   AUC 0.700).
2. **La mayor parte de ese skill es geometría, correctamente cotizada por el mercado.**
   La feature individual "distancia desde la entrada hasta el objetivo de break" alcanza
   AUC 0.74 por sí sola. Cuando el modelo está más seguro de que la zona rompe, el
   recorrido restante (la recompensa) es el más pequeño. Consecuencia:
3. **Hay ~cero trades con EV positivo tras costes realistas.** El backtest de
   sensibilidad a costes (spread 0.04 ATR + slippage 0.02 ATR, filtrado por EV con payoff
   por evento) encuentra 0 trades de break y 7 de bounce en el OOS de 1H. La salida
   honesta del motor son *probabilidades*, y las probabilidades dicen: casi nunca apostar
   con estas definiciones exactas de barreras.
4. **El modelo de bounce — la pregunta realmente operable con 2R:1R fijo — no tiene
   skill.** AUC ≈ 0.49–0.52 en ambos timeframes, BSS negativo, y **ninguna familia de
   features lo arregló**. Se entrega marcado `live_eligible: false` y la UI muestra su
   tasa base etiquetada como "not a probability".
5. **Experimentos fallidos, reportados según §11 G4:** Familia P
   (Bayes/Markov/cópulas/EVT) — sin edge medible. Familia Q de procesamiento de señal
   (FFT/wavelet) — sin edge (la frecuencia instantánea de Hilbert es la única pista:
   +0.009..0.012 BSS en bounce 1H, insuficiente para rescatar el modelo). Familia Q
   cuántico-traducida Q1–Q5, Q7, Q8 — plana a negativa, exactamente la degradación
   walk-forward que advierte la literatura QCML. **Q6 (energía de oscilador en zona) es
   el único superviviente de la rama cuántica** y una feature fuerte de break. Q9 y EMD
   no se implementaron (opcionales; su precondición — que la rama pasara su gate — nunca
   ocurrió).

## Ablación final (XAUUSD 1H, 3.436 eventos, un mismo conjunto de filas, acumulativa)

| paso | BSS bounce | AUC bounce | BSS break | AUC break |
|---|---|---|---|---|
| baseline (13 features v1) | −0.009 | 0.514 | −0.004 | 0.529 |
| +Familia P | −0.010 | 0.508 | +0.001 | 0.535 |
| +Q señal | −0.016 | 0.499 | +0.001 | 0.540 |
| +Q cuántico | −0.010 | 0.496 | **+0.013** | 0.577 |
| +Z fuerza de zona | −0.008 | 0.502 | **+0.062** | 0.655 |

Feature sets retenidos en vivo (fijados globalmente, sin snooping por par —
`model_trainer_v2.LIVE_FEATURE_RECIPE`): bounce = baseline + `hilbert_inst_freq`;
break = baseline + Z1–Z8 + `q6_harmonic_oscillator_energy` + `break_target_dist_atr`.

## Modelos finales

| | 1H bounce | 1H break | 4H bounce | 4H break |
|---|---|---|---|---|
| eventos | 4.677 | 3.882 | 1.195 | 999 |
| tasa base | 24,5% | 35,6% | 23,8% | 40,5% |
| Brier OOS | 0.186 | 0.194 | 0.182 | 0.213 |
| BSS OOS | −0.004 | **+0.153** | −0.001 | **+0.116** |
| ROC-AUC | 0.493 | **0.740** | 0.517 | 0.700 |
| ECE (10 bins) | 0.010 | 0.040 | 0.013 | 0.037 |
| C elegido (sweep P4) | 0.03 | 0.03 | 0.03 | 0.1 |
| check de cobertura conformal | 98,6% ✓ | 88,4% ✓ | 100% ✓ | **82,5% ✗** |
| elegible en vivo | **no** | sí | **no** | sí |

La LogReg L1 batió a los cinco benchmarks calibrados (XGBoost, RF, GradientBoost,
AdaBoost, GaussianNB) en Brier de break bajo el protocolo idéntico; ninguno se acercó a
la regla de reemplazo de 0.005 (`benchmarks_v2.json`).

## Limitaciones conocidas (todas verificadas, ninguna cosmética)

- **Los intervalos conformal son casi vacuos.** La no-conformidad |p − y| de la
  especificación contra una y binaria fuerza q_hat ≈ max(p, 1−p): medido 0.65–0.79 (el
  modelo v1 en producción: 0.79). Las bandas son válidas pero de ±0.7 de ancho, y la
  puerta de EV de Kelly atada a la cota inferior queda permanentemente cerrada.
  **Recomendación:** conformal por bins de probabilidad o Venn-Abers — un cambio de
  especificación, no hecho unilateralmente.
- **La cobertura del break 4H falló (82,5% < 88%).** Expuesto como
  `interval_reliable: false` en la lectura en vivo; no confiar en la banda de 4H.
- **Skill de break ≠ edge de break.** Ver puntos 2/3 del resumen. El uso práctico de la
  probabilidad de break es como *veto* (no operar contra una zona con 70% de romper), no
  como disparador de trades.
- **E3 (cierre dentro de zona) está efectivamente muerto** (n=1 en 13,7k barras): la
  prioridad de E1/E2 más la semántica de mitigación lo subsumen. Se mantiene, se reporta.
- **El calendario de bancos centrales se entrega vacío**
  (`engine/data/external/central_bank_calendar.json`). Rellenar fechas reales de FOMC/BCE
  o las lecturas cerca de decisiones NO se suprimen.
- **La detección de zonas en vivo usa una ventana de 2.000 barras** — zonas más antiguas
  son invisibles para el motor en vivo (sí eran visibles en entrenamiento). Aceptable
  para las 5 zonas más frescas por lado; documentado en `live_engine_v2.py`.
- Las fuentes de datos D1–D3 (yields reales FRED, flujos GLD, COT) **no se construyeron**:
  ningún modelo superviviente quiso features macro (la familia de cópulas falló su gate),
  así que por YAGNI esperan hasta que algo las necesite.

## Scorecard §13 (objetivos vs. logrado, XAUUSD 1H/4H)

| objetivo | logrado |
|---|---|
| Brier OOS < 0.20 en ambos modelos | break sí (0.194/0.213); bounce numéricamente sí pero sin skill |
| BSS > 0.10 | break sí (+0.153/+0.116); bounce no |
| ROC-AUC > 0.60 | break sí (0.740/0.700); bounce no |
| cobertura 90% ≥ 88% | 1H sí; break 4H no (82,5%) |
| ablación "P + Z llevan el lift" | mitad correcto: Z lo llevó; P falló por completo |
| el frontend nunca finge una señal | sí — no_signal es el estado por defecto, todo número no-modelo va etiquetado |

## Ejecutar v2

```bash
cd engine
# eventos + labels (escribe zone_events.csv en el directorio del par)
uv run python bounce_break_labeler.py XAUUSD 1H
# ablaciones por familia (reproducen todas las tablas de arriba)
uv run python probability_engine_v2.py XAUUSD 1H
uv run python quantum_features.py XAUUSD 1H signal
uv run python quantum_features.py XAUUSD 1H quantum
uv run python zone_features.py XAUUSD 1H
# entrenamiento dual completo + todos los reportes
uv run python model_trainer_v2.py XAUUSD 1H
# lectura en vivo dirigida por eventos (CLI)
uv run python live_engine_v2.py XAUUSD 1H
```

API: `GET /api/reading_v2?symbol=XAUUSD&timeframe=1H` (estados `no_signal` / `blackout` /
`signal`). Dashboard: la tarjeta "Zone Reading (v2)" en la parte superior de la página
principal.

Por par, el entrenamiento escribe: `model_bounce.pkl`, `model_break.pkl`,
`conformal_*.pkl`, `features_v2.json`, `metrics_v2.json`, `ablation_report.json`,
`regime_slice_report.json`, `cost_sensitivity.json`, `coefficient_audit.json`,
`benchmarks_v2.json`, `reliability_diagram_{bounce,break}.png`. Los archivos v1
(`model.pkl`, `features.json`) quedan intactos — ambos motores coexisten.

## Anexo (02-07-2026): frontend centrado en el trader + bandas de incertidumbre útiles

Aplicado tras el informe del Paso 10, a petición del product owner:

- **Frontend rediseñado alrededor de la decisión.** El dashboard ahora muestra solo:
  precio en vivo (la probabilidad se quitó del ticker — era del modelo v1 sin edge),
  el gráfico SMC con zonas, la tarjeta Zone Reading, una tarjeta de contexto
  Régimen & Noticias (etiquetada "solo contexto") y ajustes. Eliminado: la parrilla
  de KPIs, la curva de equity del backtest, el histograma de probabilidades, las
  tarjetas de insights de IA, el market overview, la tabla de trades, el zoo de
  modelos, los gauges (§9.3 los prohíbe), el heatmap, la sección de feature
  importance y el "+68.0 R Total Profit" hardcodeado del sidebar. Los archivos de
  componentes permanecen en disco; solo cambió la composición de la página.
- **Bandas de calibración de Wilson reemplazan la banda conformal vacua.** La banda
  de la lectura es ahora el IC de Wilson al 90% de la frecuencia observada OOS en el
  quintil de calibración de la predicción — p.ej. bin superior del break 1H: el
  modelo dice >48%, la realidad entregó 65,2% [62,1%, 68,2%]. El q_hat conformal de
  la especificación se sigue calculando y guardando (más una variante Mondrian por
  bins, medida y solo marginalmente más estrecha — el score |p−y| contra un
  resultado binario es ancho por construcción). La puerta de EV de Kelly ahora usa
  una cota inferior con significado en lugar de estar mecánicamente cerrada.
- **Contribuciones de features por lectura (§9.2 Tarjeta 3).** Cada tier calibrado
  lista sus 6 mayores |coeficiente × valor estandarizado| con signo — haciendo
  visible el dominio de la geometría en el modelo de break en vez de implícito.
- **Probado y rechazado:** dummies de tipo de evento (E1/E2 vs E4 vs E5 como
  inputs) — dBSS +0,0002…+0,0025, muy por debajo de la puerta de +0,01 en ambos
  timeframes; las features de geometría ya lo codifican. Reportado según G4, no
  adoptado.
- **Features de sesión/hora del día — probadas, bajo la puerta, no adoptadas.**
  La codificación cíclica de la hora fue la mejor pista de todo el proyecto: bounce
  1H dBSS +0,0060 con AUC 0,494→0,544 (lo único que movió el modelo de bounce),
  break 1H dBSS +0,0073; nada en 4H. Ambas quedan por debajo de la puerta de +0,01
  que rechazó candidatos más débiles, así que no están en los modelos en vivo —
  marcadas como el PRIMER re-test cuando los datos de tick amplíen los eventos.
- **Feeds de posicionamiento macro construidos (`engine/external_data.py`, §6
  D1+D3):** yields reales DFII10 de FRED (diario, sin key) y posicionamiento COT
  de managed money en oro de la CFTC (semanal, con retraso de publicación de 4
  días por causalidad), cacheados en `data/external/` con manifest. Como features
  de MODELO fallan la puerta en todos los modelos/timeframes (dBSS
  +0,0001…−0,0054) — registrado en el docstring. Se entregan solo como CONTEXTO DE
  RÉGIMEN: la tarjeta Regime & News muestra el cambio 1d de yields reales y el
  percentil de crowding COT, etiquetados "contexto, no señal".
- **Journal de señales auto-calificado (`engine/signal_journal.py` +
  `/api/journal` + tarjeta Live Track Record).** Cada lectura SIGNAL en vivo se
  registra con sus precios de barrera congelados al momento de la señal; al
  vencer el horizonte se califica contra lo que el precio realmente hizo, y el
  dashboard reporta predicho-vs-observado ("dijo 45% → ocurrió 41%, n=63") en
  total y por tercil de probabilidad. Sin backfill del histórico de entrenamiento
  — solo cuentan señales genuinamente posteriores al entrenamiento — y las tasas
  se retienen por debajo de 20 señales calificadas. Es el sistema de alerta
  temprana de deriva de calibración y la base para meta-labeling de las entradas
  del propio trader.

## Conclusión

probt v2 hace lo que un motor de probabilidad serio debe hacer: encontró una señal bien
calibrada y estable entre regímenes (roturas de zona), demostró que la mayor parte de su
fuerza es geometría que el mercado ya cotiza, mostró que el trade de bounce con R fijo no
tiene edge detectable en 2,4 años de datos, y cableó cada uno de esos hechos — incluidos
los incómodos — en los números que ve el operador. No promete rentabilidad. Promete
probabilidades honestas, y cumple esa promesa diciendo casi siempre "no apostar".
