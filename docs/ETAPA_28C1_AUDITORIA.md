# Etapa 28C.1 — Força por bordo, unidades e derivadas transversais

## Base documental

1. Lewandowski (1996), *The Transverse Dynamic Stability of Hard-Chine Planing Craft*, equações (25)–(29), p. impressa 7; apêndice para rigidez estática.
2. Lewandowski (2004), *The Dynamics of Marine Craft*, capítulo 6, Tabela 6.2, p. impressa 369.
3. `elcf/python-openplaning` branch `dynroll`, commit `a9af693e5d5de640112d4d6003f914dc31db743f`, método `get_transverse_dynamics`. Não é versão publicada nem validada.
4. Investigação anterior do projeto, Etapa 18: reconstrução das forças normais por bordo e ressalvas na Tabela 2 de Lewandowski (1996).

## Conclusões documentadas

- `0.5 * rho * U**2 * b**2 * sin(2*tau)/(2*cos(beta_eff))` em Lewandowski (1996), eq. (27): **o primeiro 1/2 é a pressão dinâmica**, o segundo é a divisão por dois bordos na projeção da sustentação. Não excluir nenhum dos fatores.
- O **segundo termo dentro do colchete** é `1.33/4`, tal como impresso em Lewandowski (1996), eq. (27). O protótipo `dynroll` contém `1/3`. Diferença numérica: `1/3 - 1.33/4 = 1/1200` (~0,000833333). O código explicita `source_variant` e não presume que a edição de 1997 resolveu a discrepância.
- Em `phi=0`, os dois bordos têm igual força normal; a **sustentação vertical equivalente** é `2 * F_side * cos(beta)` quando a geometria é simétrica. É uma identidade da decomposição, não uma verificação experimental independente.
- `beta_deg`, `tau_deg` chegam de OpenPlaning em **graus**, mas trigonometria e derivadas utilizam **radianos**. O retorno de `derivatives()` expressa `N/rad` ou `N.m/rad`.
- `beam_mode="constant"` representa a geometria da eq. (26a) com comprimento de quilha constante. `beam_mode="cosine_projected"` reproduz a projeção de boca empregada pelo protótipo, **mas não é reprodução exata**, pois o protótipo chama novamente `get_geo_lengths()`.
- O comprimento médio molhado `lambda_w=(L_K+L_C)/(2*b)` calculado pelo OpenPlaning pode divergir da relação geométrica da eq. (26a). `state.lambda_geometry_residual` informa a divergência, sem sobrescrever `L_C` ou alterar o estado Savitsky.

## Unidades: critérios implementados

| Grandeza | Unidades internas | Origem/transformação |
|---|---|---|
| U | m/s | kn = 1852/3600 m/s |
| rho | kg/m³ | slug/ft³ para SI pela conversão de massa e volume |
| L, b, KG, LCG, LCP | m | ft = 0,3048 m |
| F | N | lbf = 4,4482216152605 N |
| K, N (momentos) | N.m | lbf.ft multiplicado por 4,4482216152605×0,3048 |
| Inércia de massa | kg.m² | slug.ft² para SI |
| phi, derivadas | rad | `beta_deg`, `tau_deg` convertidos antes da trigonometria |

**Não foram convertidos coeficientes empíricos de regressões não dimensionais**; as regressões dinâmicas de 1997/2004 com expoentes de ângulo e fatores específicos são objeto da próxima subetapa. A distinção de peso versus massa permanece obrigatória.

## Escopo e limites

- Esta entrega implementa **forças por bordo**, resposta dinâmica à banda, derivadas `Y_phi`, `K_phi`, `N_phi` e aproximação do termo estático `K_phi_static` do protótipo.
- Não implementa massas adicionadas, derivadas em velocidade `Y_v` etc., amortecimento, matrizes 3×3 nem autovalores. **Não determina estabilidade ou velocidades críticas.**
- Os sinais são os do protótipo `dynroll` e ainda precisam ser conciliados com o referencial global das equações de Lewandowski antes da matriz dinâmica.
- Não existe validação contra ensaio físico nesta entrega. A reconstrução histórica da Tabela 2 de 1996 apresentou uma discrepância não resolvida de ~1,95% em forças normais, que **não deve ser mascarada por ajuste de densidade**.
- A rigidez estática é a aproximação semiempírica do protótipo, **não** uma curva GZ ou verificação de norma.
- Modelo presume casco simétrico, hard-chine e aproximação linear em banda pequena. Faixas de calibração empírica e efeitos de apêndices não são implementados aqui.

## Exemplo de uso

```python
from openplaning import PlaningBoat
from openplaning.stability import from_planing_boat, RollForceModel

# boat construído com os parâmetros usuais do OpenPlaning
boat.get_steady_trim()     # não é chamado automaticamente pelo módulo de estabilidade
boat.get_forces()          # assegura a existência de lcp

state = from_planing_boat(boat)
print("geometry discrepancy:", state.lambda_geometry_residual)
model = RollForceModel(state, source_variant="lew1996_eq27")
result = model.derivatives(step_rad=1e-5)
print("Y_phi [N/rad]:", result.Y_phi_dynamic)
print("K_phi_dynamic [N.m/rad]:", result.K_phi_dynamic)
print("K_phi_static [N.m/rad]:", result.K_phi_static)
print("N_phi [N.m/rad]:", result.N_phi_dynamic)
```

Não usar `K_phi_total` isoladamente como critério de estabilidade global.

## Próximos passos da 28C

1. Transcrever/contrastar as tabelas de derivadas hidrodinâmicas de 1997 e 2004, com unidades e definições de variáveis.
2. Resolver a convenção do produto de inércia físico `I_xz` (tensor = `-integral(xz dm)`) versus coeficientes das equações de Lewandowski.
3. Montar sistema linearizado sway-roll-yaw com `MassProperties` e resultados discriminados por variante.
4. Comparar autovalores da matriz de estado com determinante do polinômio característico.
5. Testar regressões contra valores publicados sem ajustar parâmetros para reproduzi-los.
