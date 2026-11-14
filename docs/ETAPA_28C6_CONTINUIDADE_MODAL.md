# Etapa 28C.6 — Continuidade modal e sensibilidade de Ixz

## Escopo e proveniência

Base de integração: `AltamiroAJ/python-openplaning`, branch `feature/mass-properties`, commit `bec75643` (28C.5). Código anterior de `transverse.py`, `frame_resolved.py`, `mass_properties.py` e `PlaningBoat` **não é alterado**. Os testes deste pacote são novos e independentes.

Esta análise é um diagnóstico do **modelo FSD opt-in**, com sua projeção das forças de 1996 ainda não validada experimentalmente. **Não é aprovação de estabilidade, segurança nem demonstração da identificação física dos modos.**

## Conclusão comprovada pelos três pontos do usuário

| Ixz tensor FPU (kg m²) | Re ramo baixa freq (s⁻¹) | f baixa (Hz) | Re ramo alta freq (s⁻¹) | f alta (Hz) | ramo dominante |
|---:|---:|---:|---:|---:|---|
| -144 | -3,147979 | 0,624 | -5,203 | 1,269 | baixa frequência |
| 0 | -2,094716 | 0,561 | -4,513 | 0,977 | baixa frequência |
| +144 | -4,307 | 0,608 | -3,754238 | 1,182 | alta frequência |

Esses valores foram **informados pelo usuário** para um caso real do seu modelo resolvido (1600 kg, 12 m/s); não são revalidados por este pacote sem os demais parâmetros exatos do mesmo arquivo local.

- O ramo dominante maximiza `Re(lambda)` (o menos negativo), nunca minimiza.
- O cruzamento da **parte real** entre 0 e +144 é confirmado pelos três pontos, mas isso **não estabelece** cruzamento/coalescência dos autovalores no plano complexo.
- A assimetria de respostas `Ixz = ±144` combina dependência não linear da matriz de estado e possível troca de ramo dominante; a decomposição precisa ser verificada ao longo de uma varredura.

## Formulação implementada

Estado `[v, phi, p, r]`, com `p=dot(phi)` e matrizes do modelo FSD 28C.5. Para cada `Ixz` físico (entrada `I[0,2]` do tensor **FPU** em kg m²):

1. Verifica existência de modelo de massas associado, ausência de assimetrias incompatíveis e admissibilidade física do **tensor completo** (sem admitir tensores impossíveis por ajuste de um único produto).
2. Reconstrói `RigidInertia` com `Ixz` variado, mantendo os demais termos de massa, geometria e coeficientes constantes.
3. Recalcula a matriz 4×4 e seus quatro autovalores usando `assemble_lewandowski_2004_fsd`.
4. Seleciona um autovalor com parte imaginária positiva de cada par conjugado. Não presume que a ordem retornada por `numpy.linalg.eig` seja constante.
5. Associa os dois ramos entre pontos consecutivos minimizando custo de proximidade espectral e sobreposição de autovetores direitos (`MAC` de formas **adimensionais**):

   `cost = 0.5 * |λ_new − λ_old| / max(1,|λ_new|,|λ_old|) + 0.5*(1 − |v_oldᴴ v_new|²)`.

   Escalas da forma modal para o acompanhamento: `[v/U, phi, p*b/U, r*b/U]`. Autovetores são normalizados e alinhados em fase. O custo é uma escolha numérica explícita, não lei física.
6. Calcula `dominant_branch = argmax(Re(lambda))` e localiza intervalos onde `Re(lambda_0) − Re(lambda_1)` muda de sinal. Refina por bisseção, com associação modal em cada ponto.
7. Retorna distância complexa entre ramos, número de condição da base de autovetores completa e margem entre custos de atribuição. Esses valores são diagnósticos de possíveis ambiguidades, não provas de *exceptional point* nem *avoided crossing*.

## Participação modal

Os autovetores fornecem a forma dinâmica do **modelo linear**. Há duas medidas distintas:

- `dimensionless_shapes`: componentes adimensionais `v/U, phi, p*b/U, r*b/U`, úteis à continuidade numérica. A escala adotada interfere na comparação dos módulos das componentes.
- `kinetic_fractions`: proporções aproximadas de `m |v|²`, `Ixx |p|²` e `Izz |r|²`, normalizadas (sway/roll/yaw). **Não** incluem energia potencial, massa adicionada, termos cruzados de inércia nem formas modais esquerdas. Assim, `roll_rate` é um rótulo descritivo da parcela cinética dominante, não uma identificação experimental de modo.

Para análise completa de participação e condicionamento de autovalores em matrizes não normais, deverão ser considerados autovetores esquerdos, pareamento biortogonal e sensibilidade a parâmetros. Não se declara que um ramo seja fisicamente *roll* apenas por sua frequência.

## API e execução

Com `boat` já resolvido por OpenPlaning, incluindo massas associadas e mesmas hipóteses do caso anterior:

```python
import numpy as np
import csv
from openplaning.stability import analyze_ixz_fsd

ixz = np.linspace(-144, 144, 289)  # incrementos de 1 kg m²
sweep, crossings = analyze_ixz_fsd(boat, ixz)

for crossing in crossings:
    print('Estimativa Ixz:', crossing.ixz_tensor_fpu)
    print('Intervalo:', crossing.bracket)
    print('Separação complexa:', crossing.complex_gap_per_second)

for i in (0, 144, 288):
    print(ixz[i], sweep.eigenvalues[i], sweep.dominant_branch[i],
          sweep.kinetic_fractions[i], sweep.eigenvector_condition[i])

with open('28c6_sweep_ixz.csv', 'w', newline='', encoding='utf-8') as f:
    rows = sweep.as_rows()
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
```

A função `analyze_ixz_fsd` retorna a varredura e uma tupla de cruzamentos refinados. Se algum ponto deixar de ter dois pares oscilatórios, a execução **falha de modo explícito** em vez de continuar silenciosamente com identificação de ramo incorreta. Nesse caso, a próxima implementação deverá rastrear todos os quatro autovalores, inclusive raízes reais.

## Interpretação da troca de dominância

O cruzamento de `Re(lambda)` indica que a taxa de crescimento/decadência predominante mudou de ramo. Não demonstra, por si, que duas frequências se cruzaram. Um *avoided crossing* exige analisar distância espectral completa, acoplamento e continuidade dos autovetores em parâmetro mais refinado. Em uma matriz real, não simétrica e não normal, proximidade e troca de natureza modal podem ser mais delicadas que em problemas Hermitianos.

## Testes e limite da verificação

- Casos analíticos com dois pares conjugados cujas partes reais se cruzam, sem coalescência dos autovalores.
- Continuidade dos autovetores com normalização dimensional, alinhamento de fase e associação por sobreposição.
- Dominância por máximo da parte real, exportação CSV, validação de pontos físicos de inércia e nenhuma alteração no `boat`.
- `python -m unittest discover -s tests -p 'test_modal_tracking.py' -v` e `python -m unittest discover -s tests -p 'test_*.py' -v`.

A referência exata `−3,120900` da 28C.3 e os valores FSD enviados pelo usuário dependerão da configuração completa de massa, geometria e hidrodinâmica no computador dele, que não foi reconstruída aqui. Verificar a saída com os scripts já preservados em `04_Modelo_Analise/codigo/`.
