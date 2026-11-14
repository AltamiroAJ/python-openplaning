# Etapa 28C.11a — Auditoria documental do fator de molhamento lateral

Base: `feature/mass-properties`, commit `79017622` (28C.10). **Nenhum arquivo de código de produção foi alterado.**

## Leitura da fonte e falso positivo

A Tabela 6.2, p. 369, de E. M. Lewandowski, *The Dynamics of Marine Craft* (2004), apresenta, no ramo de baixa velocidade:

`f_sw = 1 + 0.603 L_C2 / (lambda B)`.

O símbolo `lambda B` está inteiramente no denominador. A interpretação alternativa `1 + 0.603 L_C2 lambda/B` foi um **falso positivo de transcrição**. Não constitui defeito do código. A função `_source_fsw` em `openplaning/stability/source_audit.py` já reproduz a expressão impressa.

A distinção não pode ser decidida por análise dimensional, porque ambas as expressões são adimensionais. A fonte foi relida na imagem ampliada (600 dpi) e cotejada com reproduções numéricas independentes reportadas pelo pesquisador. Não inserir como valores-alvo os números da transcrição invalidada (1.12526, 1.75474, 2.22214), nem os respectivos deltas de `K_phi_static`.

## Convenção na fronteira

O código aplica:

- se `U**2 > g*B*(lambda - 0.16*tan(beta)/tan(tau))/(3*sin(tau))`: `f_sw = 1`;
- em caso contrário, inclusive **igualdade**: `f_sw = 1 + 0.603*L_C2/(lambda*B)`.

**A escolha na igualdade é uma convenção do programa.** O texto-fonte utiliza desigualdades estritas e não define expressamente essa fronteira. O teste com igualdade exata usa uma parametrização matemática sintética e funções trigonométricas controladas para eliminar ambiguidades de ponto flutuante. Não representa barco fisicamente admissível.

Essa convenção pode gerar um salto finito em `f_sw` no limiar se `L_C2 != 0`; nenhum alisamento ou clamp foi introduzido.

## Regressão independente

Para o fixture sintético da 28C.7 (`B=2.2 m`, `L_K=6.051 m`, `L_C=5 m`, `L_C2=0.2 m`, `rho=1025 kg/m³`, `g=9.81 m/s²`, `beta=18°`, `tau=3.0916°`, `KG=0.65 m`):

| Resultado | Valor literal esperado |
|---|---:|
| `lambda` | 2.5115909090909088 |
| `f_sw` no ramo baixo | 1.0218260790878653 |
| `K_phi_static_2004` no ramo baixo (N·m/rad) | -24235.07487158906 |
| `f_sw` no ramo alto | 1.0 |
| `K_phi_static_2004` no ramo alto (N·m/rad) | -23717.416659812145 |

São referências de regressão de **um fixture sintético**, não de um caso físico de 16 kN/10 m/s. Os valores de baixa velocidade registrados para os três casos reais e conferidos pelo pesquisador permanecem: aproximadamente 1.03707, 1.18796 e 1.26456. Não os usar como testes unitários sem os estados resolvidos correspondentes.

A preservação do R6 real em `1600 kg/12 m/s` é sustentada pela execução anterior com `f_sw=1` e pelo fato de **não haver alteração no código de produção**. A suíte automatizada deste pacote não reconstitui aquele `PlaningBoat` real; se for necessária nova evidência executável, rodar os scripts R6 existentes no ambiente do projeto.

## Escopo e validade

Teste novo: `tests/test_fsw_28c11a_documental.py` (7 casos): valor literal baixo, valor literal alto, propagação a `K_phi_static`, distinção `L_C2/L_C`, igualdade exata na fronteira e ausência de mutação. Arquivos `source_audit.py`, `lewandowski.py`, `transverse.py`, `source_labeled.py` e demais solvers permanecem inalterados. Nenhum coeficiente numérico foi modificado. A aprovação documenta fidelidade textual e regressão da implementação existente; não constitui validação física do modelo.
