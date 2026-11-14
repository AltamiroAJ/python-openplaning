# Etapa 28C.3 — Sistema sway–roll–yaw e autovalores (experimental)

## Origem e validação documental

- Lewandowski (2004), *The Dynamics of Marine Craft*, cap. 6, pp. 362–365, Eq. 6.8 (inércia rígida), 6.9 (derivadas), 6.10 (equações diferenciais), 6.12 (matriz quadrática/linear), 6.13 (polinômio de ordem 4).
- Derivadas hidrodinâmicas: módulo 28C.2, variante `lew2004_table63` exclusivamente.
- Derivadas de banda: módulo 28C.1, variante `lew1996_eq27` exclusivamente, ressalvadas as convenções ainda não validadas.
- **Não** usamos a expansão manual dos coeficientes Eq. 6.14, cujo sinal de alguns termos foi questionado na auditoria precedente. O determinante polinomial é calculado diretamente da Eq. 6.12, e conferido independentemente contra `numpy.poly(A)`.

## Formulação

O vetor de estado é `x = [v, phi, p, r]`, com `p = d(phi)/dt` e `r = d(psi)/dt`.

Definimos `q = [v,p,r]`. A Eq. 6.10 é:

`M * q_dot + D * q + h * phi = 0`

com

```
M = [[Y_vdot - m,         Y_phiddot,          Y_rdot       ],
     [K_vdot,             K_phiddot - Ixx,    K_rdot + Ixz ],
     [N_vdot,             N_phiddot + Ixz,    N_rdot - Izz ]]

D = [[Y_v, Y_phidot, Y_r - m*U],
     [K_v, K_phidot, K_r],
     [N_v, N_phidot, N_r]]

h = [Y_phi, K_phi_dynamic + K_phi_static, N_phi]
```

O membro do livro `Ixz` foi assumido **produto bruto** `integral(x*z dm)`, numericamente `-inertia_tensor[0,2]` no tensor físico. A identificação do sinal entre a convenção do livro, que utiliza uma matriz de momentos de inércia, e o referencial da 28B permanece uma **hipótese explícita de mapeamento**, não uma correspondência experimentalmente confirmada. A transformação de eixos do livro para o frame `x` vante, `y` bombordo, `z` cima ainda exige revisão dos sinais das derivadas antes de conclusões físicas.

`A[1,2] = 1`, `A[[0,2,3],:] = -solve(M, [D_v,h,D_p,D_r])`.

O polinômio é `det(P(s))` onde a segunda coluna de `P(s)` contém `M[:,1] s² + D[:,1] s + h`; as outras colunas possuem grau 1. O código compara seus coeficientes, normalizados, com `numpy.poly(A)` e seus zeros com `eigvals(A)`.

## Dimensões SI

| Quantidade | Unidade |
| --- | --- |
| v, U | m/s |
| phi | rad |
| p, r | rad/s |
| Ixx, Izz, Ixz | kg·m² |
| Y, K, N | N; N·m; N·m |
| Re(s) | 1/s |
| |Im(s)|/(2pi) | Hz |

β e τ permanecem em **graus somente nas regressões originais**, como detalhado em 28C.2. Os derivados angulares da 28C.1 são por radiano.

A matriz `M` possui unidades heterogêneas. O teste de condicionamento utiliza escalas explícitas de massa e comprimento; o `cond(M)` numérico sem escalonamento não é fisicamente interpretável.

## Limites e hipóteses

- Não há cálculos automáticos de equilíbrio: `boat.get_steady_trim()` e `boat.get_forces()` devem ter sido chamados previamente. Não há verificação de convergência nesta etapa; o usuário deve conferir os resíduos.
- `boat.get_mass_properties_snapshot()` deve existir e aceitar o modelo de massa em referencial consistente, com carregamento lateral simétrico.
- Propriedades rígidas derivam de massas físicas; nenhuma massa adicionada é computada a partir do deslocamento.
- Não há contribuições de apêndices, propulsão lateral, ventilação ou fenômenos não lineares.
- `1997` e `dynroll` ficam bloqueadas na montagem: a primeira por inconsistência dimensional identificada em 28C.2; a última por não ser publicada nem validada.
- A hipótese do comprimento de referência `L=(L_K+L_C)/2` nas regressões 28C.2 e a geometria de casco molhado nos esforços 28C.1 permanecem sujeitas a validação.
- O resultado `stable_linear_model` ou `unstable_linear_model` **classifica apenas o sistema matemático montado**, nunca fornece certificação de embarcação ou validação física de comportamento em mar real.

## Uso

```python
from openplaning.stability import from_resolved_planing_boat

# Depois de equilíbrio longitudinal e associação de MassProperties:
result = from_resolved_planing_boat(boat)
print(result.classification)
print(result.eigenvalues)
print(result.spectral_abscissa, '1/s')
print(result.oscillation_frequencies_hz, 'Hz')
print(result.acceleration_matrix)
```

Para reproduzir equações matriciais isoladas, `linearize_transverse_system(M,D,h,mass=...,beam=...)` recebe diretamente matrizes e retorna autovalores, polinômio e indicadores.

## Testes

- 21 novos testes específicos, incluindo quatro raízes analíticas desacopladas, caso instável, marginalidade, consistência entre polinômios independentes, sinal de `Ixz`, invariância dos insumos e rejeição de variantes.
- 57 testes cumulativos nos módulos extraídos 28C.1, 28C.2 e 28C.3, executados neste ambiente isolado.
- **Não** foram executados no ambiente isolado os testes completos do fork e a integração com uma instância real de `PlaningBoat` do commit remoto; devem ser executados localmente após instalação.

## Próxima etapa 28C.4

1. Auditar os referenciais exatos do capítulo 6 (sinais Y, K, N, phi, r) e o termo cruzado `Ixz` para coordenadas físicas.
2. Avaliar alterações de `L`, ensaios de Brown e apêndices antes de qualquer avaliação quantitativa de segurança.
3. Integrar resultados à fachada pública com testes do `PlaningBoat` real e estados de equilíbrio controlados.
