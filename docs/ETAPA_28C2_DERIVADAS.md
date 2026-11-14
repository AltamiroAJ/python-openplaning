# Etapa 28C.2 — Derivadas hidrodinâmicas de Lewandowski

## Fontes primárias verificadas

- E. M. Lewandowski, *The Transverse Dynamic Stability of Planing Craft*, Marine Technology, abril de 1997, apêndice *Evaluation of Hydrodynamic Coefficients and Stability Criterion*, p. 117 (tabela de coeficientes de casco).
- E. M. Lewandowski, *The Dynamics of Marine Craft: Maneuvering and Seakeeping*, 2004, cap. 6, tabelas 6.2 e 6.3, pp. 369–370.
- Repositório `elcf/python-openplaning`, branch `dynroll`, método experimental `get_transverse_dynamics()`.

## Escopo e limites

O arquivo `openplaning/stability/derivatives.py` fornece 18 derivadas hidrodinâmicas *do casco* (acelerações, velocidades e taxas de banda). Não inclui massas rígidas, massas adicionadas dos apêndices, termos restauradores, equações acopladas ou autovalores. Não altera `PlaningBoat` ou a solução de equilíbrio longitudinal.

Dados SI: U [m/s], rho [kg/m³], g [m/s²], B, L_K, L_C, T, KG, LCG [m]. **T é o calado na popa já calculado por OpenPlaning (`boat.T`)**, não uma estimativa arbitrária. O parâmetro longitudinal `L` foi definido explicitamente como `(L_K + L_C)/2` para as regressões de `Y_r` e `N_r`; a identificação deve ser confirmada contra a definição original do comprimento usado nos ensaios de Brown–Klosinski.

As regressões `β**0.6`, `β**0.342` e `(τ+5.28)` preservam **graus**, enquanto funções trigonométricas recebem ângulos convertidos para **radianos**. O `C_V²` de `Y_v` na tabela publicada é mantido. Não há conversão global de coeficientes dimensionais imperiais: as expressões escolhidas são reexpressas diretamente em grandezas SI e verificadas por homogeneidade.

## Variantes

- `lew2004_table63`: tabela 6.3, incluindo `N_r` com `+6.714 T'²/(τ+5.28)²`, `K_phi_dot` com `h1*Y_v`. Todos os termos implementados são dimensionalmente coerentes.
- `lew1997_appendix`: apêndice de 1997, incluindo `N_r` com `-0.08289 T'²/(τ+5.28)²` dentro de fator externo negativo. A publicação apresenta `K_phi_dot` com `h1*Y_v/U`, cuja unidade `[kg·m]` não pode ser somada diretamente ao termo principal `[kg·m²/s]`. O resultado histórico é calculado, mas `require_dimensional_consistency()` o rejeita; não usar para estabilidade até resolver a fonte.
- `dynroll_prototype`: variante diagnóstica reproduzindo desvios do código incompleto: falta `C_V²` em `Y_v`, usa coeficiente 2 (em vez de 3) no termo `L_C² L_K` de `N_r_dot`, e reproduz a expressão de `N_r` do protótipo. A variante fica bloqueada para integração automática por ser experimental, mesmo quando as dimensões são compatíveis.

Não se afirma que o termo `h1*Y_v/U` seja erro autoral definitivo. Pode representar ambiguidade de símbolo ou transcrição; requer confronto com documentação experimental original. Não se misturam fontes silenciosamente.

## Convenções e grandezas

`v` velocidade lateral [m/s], `r` velocidade angular de yaw [rad/s], `phi` roll [rad]. `Y` força [N], `K`, `N` momentos [N·m]. Radianos são adimensionais no SI, mas identificados semanticamente nas derivadas.

- `Y_v_dot`: kg; `N_v_dot` e `Y_r_dot`: kg·m; `N_r_dot`: kg·m².
- `Y_v`: kg/s; `N_v`, `K_v`, `Y_r`: kg·m/s; `N_r`, `K_phi_dot`: kg·m²/s.
- `K_phi_ddot`: kg·m².

## Uso

```python
from openplaning.stability import coefficient_state_from_planing_boat, evaluate_hull_derivatives

# boat.get_steady_trim() e boat.get_forces() devem ter sido chamados antes.
state = coefficient_state_from_planing_boat(boat)
result = evaluate_hull_derivatives(state, source='lew2004_table63')
result.require_dimensional_consistency()
print(result['Y_v'], result.units['Y_v'])
print(result['N_r'], result.units['N_r'])
```

## Testes

- 16 novos testes unitários: cobertura, validação de entradas, simetria de massas adicionadas, regressões independentes, escala em massa específica, velocidade e similaridade de Froude, rastreamento das variantes e extração sem mutação.
- 20 testes da etapa 28C.1 executados novamente em ambiente local isolado: total 36/36.
- **Não** foram executados os testes completos de `PlaningBoat` da versão remota neste ambiente. Executá-los no fork após a aplicação.

## Próxima etapa 28C.3

Confirmar referencial de sinais e origem das acelerações; montar `M`, `C`, `K` e a matriz de estado sway-roll-yaw usando `MassProperties` explícito. A variante de 1997 não deve alimentar o solver enquanto persistir a incompatibilidade dimensional.
