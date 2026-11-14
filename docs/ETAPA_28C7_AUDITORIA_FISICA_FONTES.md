# Etapa 28C.7 — Auditoria física dos bordos e coeficientes de banda

**Status:** auditoria diagnóstica, com discrepâncias demonstradas. **NÃO encerrada a validação física.** Não alterar o solver 28C.3/28C.5 nesta etapa; não substituir automaticamente coeficientes por valores desta auditoria.

## Fontes primárias inspecionadas diretamente

1. Lewandowski, E. M. (1996), *The Transverse Dynamic Stability of Hard-Chine Planing Craft*, pp. 6–8, eqs. (23)–(31): força normal por bordo, β efetivo, comprimento molhado dependente de β, braço de momento, restauração estática e derivação por ângulo de banda.
2. Lewandowski, E. M. (2004), *The Dynamics of Marine Craft*, capítulo 6, pp. 369–370, Tabelas 6.2 e 6.3: `F_phi`, `Y_phi`, `K_phi`, `N_phi`, `Delta`, `KB`, `f_sw`. O próprio livro declara **τ em radianos na expressão `F_phi`**, enquanto outras regressões empíricas de suas tabelas usam graus.
3. Lewandowski (1997), *The Transverse Dynamic Stability of Planing Craft*, apêndice p. 117, para comparação documental de variantes do casco. Não usado para compor a nova derivada de banda.
4. Código publicado no fork `AltamiroAJ/python-openplaning`, branch `feature/mass-properties`, commit base `fd32b681`: `lewandowski.py`, `derivatives.py`, `transverse.py`, `frame_audit.py`, `frame_resolved.py`, `modal_tracking.py`.

## Matriz de rastreabilidade de equações

| Grandeza | Fonte | Código auditado | Resultado |
|---|---|---|---|
| Força normal por bordo F(β±φ) | 1996 eqs. (25)–(27) | `RollForceModel.side_normal_force` | A expressão de base segue eq. 27 com coeficiente 1.33/4; mantém geometria particular baseada em eq. 26 |
| K_Dφ = d[(F_port−F_stbd)h₂]/dφ | 1996 eqs. (28)–(30) | `RollForceModel.forces_at_heel` | Identidade diferencial verificada com derivada de cinco pontos |
| `F_phi` | 2004 Tabela 6.2 | `audit_roll_sources`: expressão impressa independente | **Não é igual numericamente** à derivada da expressão de 1996 em todos os estados; não identificar silenciosamente |
| Y_φ = 2F_φ sinβ | 2004 Tabela 6.3 | `audit_roll_sources` | **Difere** de diferenciar a projeção de duas forças normais adotada em `lewandowski.py` |
| K_Dφ = 2h₂F_φ | 2004 Tabela 6.2 | `audit_roll_sources` | Fórmula calculada com F_φ impresso, distinguindo-a de K da eq. 1996 |
| N_φ = Y_φ(LCP−LCG) | 2004 Tabela 6.3 | `audit_roll_sources` | Ambas as convenções produzem N_φ distintos se seus Y_φ forem distintos |
| Δ, KB e parcela estática K_φ | 2004 Tabelas 6.2–6.3 | `RollForceModel.static_roll_derivative` | **Divergência numérica**: os fatores, aproximações e a condição para f_sw não coincidem integralmente |
| `f_sw` condicionado à velocidade | 2004 Tabela 6.2 | `_source_fsw` | Valor condicional rastreável; `RollForceModel` não usa a mesma condição |
| `L_C` e λ durante banda | 1996 eq. (26) | `RollForceModel._geometry` | Diferenciação usa eq. 26 para lados com L_K fixo; geometria original de Savitsky pode ter `L_C` inicial não coincidente |
| FPU→FSD, `Ixz` | 2004/28C.4 | `frame_resolved.py` | **Não alterado** aqui. Necessita fechamento de sinais físicos completo |

## Discrepância 1 — Diferenciação da projeção lateral

Para as duas forças normais e o sinal usado na implementação de 1996:

`Y(φ) = F(β+φ) sin(β+φ) − F(β−φ) sin(β−φ)`.

Logo, **por regra da cadeia**, em φ=0:

`dY/dφ = 2 F′(β) sinβ + 2 F(β) cosβ`, onde `F′` denota a derivada por radiano da força normal de *um bordo*.

Já a expressão publicada em 2004, Tabela 6.3, é `Y_φ = 2 F_φ sinβ`, com `F_φ` especificado na Tabela 6.2. Existem **duas diferenças** a conciliar: (a) a expressão de F_φ da Tabela 6.2 não é numericamente idêntica à derivada obtida da versão de 1996; (b) a tabela de 2004 não apresenta o termo `2 F(β) cosβ` que decorre da diferenciação da projeção completa selecionada em `RollForceModel`.

**Isto não comprova erro do livro**: a quantidade Y_φ pode ter definição/convenção de projeção distinta; precisa ser conferida contra o desenvolvimento original e eventuais simplificações da linearização. O erro que já pode ser afirmado é chamar as duas definições de equivalentes sem demonstrar a equivalência.

## Discrepância 2 — Restauração estática e molhamento

A Tabela 6.2 (2004) fornece:

- `Δ = 0.25 ρ g λ² sin(2τ) B³` (N)
- `KB = B tanβ (0.5 + L_C/L_K)/6` (m)
- `f_sw = 1` acima do limiar indicado na tabela; abaixo, `f_sw = 1 + 0.603 L_C2/(λB)`
- `K_φ,est = 0.624 f_sw[−ρg B³(L_K+3L_C)/48 + (KG−KB)Δ]` (N·m/rad)

O código prévio `static_roll_derivative()` emprega expressão de Δ distinta (inclui fator corretivo adicional), não aplica explicitamente a condição de velocidade para f_sw e portanto não reproduz literal e universalmente as Tabelas 6.2/6.3. A **equação (23) de 1996 é também uma variante histórica distinta**: não se deve misturar sua definição de Δ com a expressão de 2004.

**Observação:** a notação tipográfica de limiar `g B (λ−0.16 tanβ/tanτ)/(3 sinτ)` foi interpretada com `3 sinτ` como denominador, coerente com a condição de molhamento por comprimento. Esse agrupamento e a igualdade exata no limiar deverão ser confrontados com o desenvolvimento da fonte para validação física. A implementação evita divisão por zero porque `τ>0` é exigido na entrada.

## Exemplo estritamente ilustrativo (não reproduz o caso real do usuário)

Parâmetros do teste isolado: `U=12 m/s`, `m` não usado, `ρ=1025 kg/m³`, `B=2.2 m`, `β=18°`, `τ=3.0916°`, `L_K=6.051 m`, `L_C=5.0 m`, `L_C2=0.2 m`, `KG=0.65 m`, `LCG=2.25 m`, `LCP=2.6 m`.

- Força normal por bordo em φ=0 (1996): **8269.766 N**.
- `F′` (derivada da eq. 27/1996): **−14216.892 N/rad**.
- `F_φ` impresso na Tabela 6.2/2004: **−13434.506 N/rad**.
- `Y_φ` da projeção completa do código anterior: **+6943.507 N/rad**.
- `Y_φ` da Tabela 6.3/2004 calculado com `F_φ` impresso: **−8302.981 N/rad**.
- Termo exclusivamente geométrico `2F(0)cosβ`: **+15730.030 N/rad**.
- Restauração estática do livro, variante 2004: **−24235.075 N·m/rad**; `static_roll_derivative()` legado: **−24217.874 N·m/rad**.
- Resíduo geométrico `lambda_w − lambda_eq26`: **+0.718581** (adimensional). A diferença deve ser investigada em cada estado real antes de confiar nas derivadas.

O valor da abscissa `−2.094716 s⁻¹` da 28C.5 não deve ser alterado por este pacote, que **não integra** as fórmulas auditadas ao solver.

## Critérios de encerramento (ainda NÃO satisfeitos)

1. Resolver a definição física de `Y_φ` e `F_φ` entre 1996 e 2004; escolher uma variante de fonte única, com justificativa, sem combinar indevidamente expressões.
2. Confirmar o papel de `L_C`, `L_K` e `λ` no estado de equilíbrio real e na perturbação por banda; tratar o resíduo geométrico.
3. Decidir explicitamente o emprego das expressões de restauração estática de 1996 **ou** 2004 e a condição de `f_sw`.
4. Confrontar sinais de `Y,K,N`, `p`, `r` e `φ` com 2004 eqs. 6.8–6.12 no referencial FSD; não fazer mudanças de sinais isoladas.
5. Reexecutar regressões do caso de 1600 kg / 12 m/s com uma variante física selecionada e justificativa por equação, acompanhadas de incerteza de parâmetro e calibração/validação externa.

**Estado de manuscrito:** os resultados R1–R4 continuam diagnóstico de implementação, não conclusão de estabilidade física. Esta etapa encontra discrepâncias rastreáveis, **não as resolve por calibração**.

## API diagnóstica

```python
from openplaning.stability import audit_roll_sources, from_planing_boat
state = from_planing_boat(boat) # após equilíbrio + forças já resolvidos
result = audit_roll_sources(state)
print(result.y_phi_full_projection_N_per_rad)
print(result.y_phi_2004_table63_N_per_rad)
print(result.source_consistency_flags)
```

## Testes

- 22 novos testes do módulo fonte/identidades algébricas/duas variantes de F_φ/condição f_sw/estado não mutado.
- 157/157 testes locais isolados, incluindo os novos e as versões anteriores montadas no ambiente de teste; `compileall` sem erros.
- A suíte real do fork e a simulação física do caso do usuário deverão ser repetidas após aplicação manual do pacote.
