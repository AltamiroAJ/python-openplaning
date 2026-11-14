# Etapa 28C.8 — Equilíbrio, linearização e comprimentos molhados

## Base documental

1. E. M. Lewandowski, *The Transverse Dynamic Stability of Hard-Chine Planing Craft* (1996), equações (24)–(31), especialmente equações (26), (26a), (27) e (29), pp. 6–8.
2. E. M. Lewandowski, *The Dynamics of Marine Craft: Maneuvering and Seakeeping* (2004), capítulo 6, tabelas 6.1–6.3 (pp. 368–370) e discussão da restauração dinâmica (pp. 371–373), especialmente equações (6.20)–(6.21).
3. Código do fork `AltamiroAJ/python-openplaning`, `feature/mass-properties`, commit base `c5ed8557`, módulos `lewandowski.py` e `source_audit.py`.

## 1. Hipótese discriminante e sua limitação

A pergunta proposta é se a parcela `2*F0*cos(beta)` desaparece ao linearizar a **força lateral total** em torno do equilíbrio transversal.

Para o modelo de projeção por bordo implementado em 28C.1,

```
Y_d(phi) = F_port(phi)*sin(beta+phi) - F_stbd(phi)*sin(beta-phi)
F_port(0) = F_stbd(0) = F0
Y_d(0) = 0
Y_d'(0) = 2*F_phi,1996*sin(beta) + 2*F0*cos(beta)
```

Se `Y_s(phi)` for a contribuição lateral restante (estática, apêndices, propulsão, etc.), o equilíbrio exige somente `Y_d(0)+Y_s(0)=0`. A derivada total é:

```
Y_total'(0) = Y_d'(0) + Y_s'(0)
```

Portanto **equilíbrio de força em phi=0 não cancela a derivada**. A força compensadora constante `Y_s(phi)=-Y_d(0)` preserva `Y_d'(0)`; uma lei com `Y_s'(0)=-2*F0*cos(beta)` cancela apenas a parcela geométrica. Ambas satisfazem o mesmo equilíbrio. Não existe evidência, nas equações inspecionadas, de que a derivada lateral estática tenha necessariamente esse valor. A expressão escalar para lift estático `Delta` da Tabela 6.2 não determina univocamente essa derivada lateral.

Consequentemente, a hipótese de que `2*F0*cos(beta)` é **exclusivamente um termo de referência** não está demonstrada; não se deve eliminar esse termo automaticamente do Jacobiano.

O teste implementado `test_equilibrium_derivative` aceita uma derivada lateral estática **prescrita externamente**, claramente identificada como hipótese; o padrão zero NÃO afirma que o casco não possua efeitos estáticos laterais. Não é um solver de equilíbrio transversal com liberdade de heave/roll.

## 2. A questão de `L_C` não é somente numérica

`RollForceModel.side_normal_force()` recalcula a geometria de cada bordo por `L_C(phi) = L_K - b*tan(beta_eff)/(pi*tan(tau))` (1996, Eq. 26), usando `L_K` fixo e deadrise efetivo. Não utiliza diretamente `state.L_C`. Logo, variar `state.L_C` sem alterar `L_K, b, beta, tau` não altera a derivada da força normal desse módulo. Isso foi demonstrado por teste.

A Tabela 6.2 de 2004 utiliza `lambda=(L_K+L_C)/(2*b)` com o comprimento fornecido. Se `L_C` do Savitsky não satisfizer a Eq. 26a, comparar as duas expressões significa comparar **hipóteses geométricas distintas**.

Em particular, `lambda_eq26 = L_K/b - tan(beta)/(2*pi*tan(tau))` e `lambda_sav=(L_K+L_C_sav)/(2*b)`.

## 3. Decomposição telescópica explícita

Definimos:

```
Y96_full = derivada numérica da projeção lateral completa (1996)
Y96_rate = 2*F_phi_1996*sin(beta)
Y04_eq = 2*F_phi_2004(lambda_eq26)*sin(beta)
Y04_sav = 2*F_phi_2004(lambda_sav)*sin(beta)
```

O resíduo `Y96_full - Y04_sav` admite a decomposição exata:

```
R_projecao  = Y96_full - Y96_rate
R_fonte     = Y96_rate - Y04_eq
R_geometria = Y04_eq - Y04_sav
R_total     = R_projecao + R_fonte + R_geometria
```

`R_projecao = 2*F0*cos(beta)` pela regra do produto. `R_fonte` confronta as expressões de derivada da força normal **no mesmo comprimento médio de referência**. `R_geometria` mede a substituição do comprimento molhado na fórmula tabulada de 2004.

**Cuidado:** a repartição é dependente da ordem do contrafactual; não é uma prova de causalidade física exclusiva. O componente de `L_C` não é uma "correção" física obrigatória: é o efeito de escolher a geometria de uma fonte dentro da outra.

## 4. Estado ilustrativo reproduzível (NÃO o caso exato do usuário)

`U=12 m/s, rho=1025 kg/m³, b=2.2 m, beta=18°, tau=3.0916°, L_K=6.051 m, L_C=4.65 m, VCG=0.65 m, LCG=2.25 m, LCP=2.27 m`.

| Parcela | N/rad |
|---|---:|
| Projeção | +15 730.03 |
| Diferença entre expressões a `lambda_eq26` | +36.72 |
| Comprimento molhado | −483.15 |
| Resíduo total | +15 283.60 |

No exemplo, `L_C_eq26 ≈ 1.838 m` difere de `L_C_sav = 4.650 m`. Essa discrepância depende do estado hipotético escolhido; **não substituir** pelos números do caso real de 1600 kg, cujos valores completos de geometria precisam ser lidos do `boat` resolvido.

O mesmo desdobramento para o momento `N_phi` exige multiplicar pelas distâncias longitudinais, assumindo **o mesmo braço LCP−LCG** de ambos os lados; não identifica separadamente eventuais diferenças de centros de pressão entre fontes.

## 5. APIs

```python
from openplaning.stability import (
    from_planing_boat,
    decompose_lateral_discrepancy,
    test_equilibrium_derivative,
)

s = from_planing_boat(boat)    # barco previamente resolvido, sem mutação
r = decompose_lateral_discrepancy(s)
print(r.projection_term, r.source_difference_at_matched_lambda,
      r.wet_length_difference, r.total_difference)

e = test_equilibrium_derivative(s)  # força lateral estática adicional, derivada=0 (cenário apenas)
print(e.total_force_zero_N, e.total_rate_N_per_rad)
# Contrafactual: força adicional com derivada exatamente oposta à parcela geométrica
other = test_equilibrium_derivative(
    s, static_heel_rate_N_per_rad=e.cancellation_requires_rate_N_per_rad)
print(other.total_rate_N_per_rad)
```

## 6. Parecer

- **Demonstrado:** em `RollForceModel` a força lateral resultante já é nula em banda zero por simetria, mas a derivada geométrica não é nula; equilíbrio não a elimina automaticamente.
- **Demonstrado:** 1996 e 2004 utilizam regras diferentes para o comprimento molhado na implementação atual; a diferença é isolável numericamente.
- **Não demonstrado:** a formulação tabulada de 2004 ser a derivada total da mesma força lateral não linear utilizada no módulo de 1996.
- **Não demonstrado:** a contribuição lateral estática real cancelar a parcela geométrica; a fonte não oferece a função lateral estática completa `Y_s(phi)` para provar essa identidade.
- **Não autorizado:** alterar `Y_phi`, `N_phi`, `K_phi` do solver, reinterpretar os autovalores como resultados validados ou declarar as duas fontes fisicamente equivalentes.

**Pendência bloqueante permanece aberta.** Para encerrá-la, exige-se a derivação física da contribuição estática lateral e a reconciliação da definição de força/momento lateral em ambas as fontes, idealmente com ensaios de força lateral em banda não nula a velocidade e trim controlados.
