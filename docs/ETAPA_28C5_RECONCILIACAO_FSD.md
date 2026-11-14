# Etapa 28C.5 — Reconciliação explícita das coordenadas do sistema sway–roll–yaw

**Base:** branch `feature/mass-properties`, commit `11c6b53` (28C.4).  
**Regra:** manter `assemble_lewandowski_2004` e `from_resolved_planing_boat` (28C.3) **inalterados** como regressões históricas. Disponibilizar uma versão reconciliada **opt-in**.

## 1. Convenções que são evidência versus hipóteses

**Documental:** Lewandowski (2004, cap. 1 e 6) utiliza sistema FSD, `x` para vante, `y` boreste, `z` baixo; a origem do subsistema é o CG. O modelo de massa OpenPlaning 28B usa FPU, `x` vante, `y` bombordo, `z` cima. A transformação espacial rígida é `R=diag(1,-1,-1)`, uma rotação própria.

**Matemática:** para tensor físico `I_tensor = ∫(r² 1 − rrᵀ) dm`, fora da diagonal `I_xz = -∫xz dm`. Portanto, para o produto bruto escrito na equação de Lewandowski:

```
P_xz,book,FSD = ∫x_FSD z_FSD dm = +I_xz,tensor,FPU
```

A antiga propriedade `RigidInertia.Ixz_book_raw` da 28C.3 retorna `-I_xz,tensor,FPU`, que é o **produto bruto FPU**, não FSD. A 28C.5 **não altera** essa propriedade antiga; sua nova montagem utiliza explicitamente `+rigid.Ixz_tensor`.

**Hipótese ainda não validada:** as derivadas de casco `lew2004_table63` são interpretadas diretamente em FSD, como publicado. As forças por bordo `RollForceModel` da 28C.1 **seguem as projeções assinadas do protótipo** e são interpretadas aqui como FPU. Embora a transformação dessas componentes seja matematicamente definida, a identificação das projeções de 1996 com forças físicas FPU **não está comprovada experimentalmente**. Esta etapa produz uma montagem com referenciais explicitamente identificados e uma hipótese interpretativa reproduzível, não um modelo físico validado.

## 2. Mudanças atômicas no caminho novo (não no legado)

O código novo cria `openplaning/stability/frame_resolved.py`. As matrizes da nova versão usam variáveis dinâmicas e linhas FSD:

```
q_FSD = [v_starboard, p_roll_forward, r_yaw_down]
w_FSD = [Y_starboard, K_roll_forward, N_yaw_down]
phi_FSD positivo ao redor de x

M[1,2] = K_r_dot + P_xz,book,FSD
M[2,1] = N_phi_ddot + P_xz,book,FSD
P_xz,book,FSD = Ixz_tensor,FPU

h_FPU = [Y_phi_dynamic, K_phi_total, N_phi_dynamic]
h_FSD = diag(-1,+1,-1) @ h_FPU
```

As derivadas de velocidade e aceleração `derivatives.py` são interpretadas como **FSD de origem**; convertê-las novamente, sem evidência de que foram calculadas em FPU, seria dupla transformação. O termo inercial `Y_r - M U` é preservado na mesma base FSD publicada. Nenhuma constante empírica, método Savitsky ou resultado de massa é alterado.

A transição para FPU, quando requisitada para inspeção, transforma **as matrizes completas** com o `DescriptorTransform` da 28C.4; ela não inverte apenas um termo. Os autovalores são invariantes sob essa transformação completa da versão 28C.5.

## 3. Interface e coexistência

```python
from openplaning.stability import (
    from_resolved_planing_boat,       # LEGADO 28C.3: intocado
    from_resolved_planing_boat_fsd,   # 28C.5: hipótese explícita FSD
    compare_legacy_and_fsd,
)

# `boat` deve ter equilíbrio geométrico e forças resolvidos por OpenPlaning,
# além de massa associada por boat.set_mass_properties(...).
legacy = from_resolved_planing_boat(boat)
reconciled = from_resolved_planing_boat_fsd(boat)
comparison = compare_legacy_and_fsd(boat)

print(comparison.legacy.spectral_abscissa)
print(comparison.fsd.spectral_abscissa)
print(comparison.spectral_abscissa_difference)
print(comparison.eigenvalue_maximum_assignment_difference)
```

As versões não modificam o objeto `boat` nem executam novamente o equilíbrio. A comparação reconstrói ambas a partir **do mesmo estado e das mesmas propriedades de massa**.

## 4. Sobre a referência −3,1209 s⁻¹

A referência do usuário corresponde a `1600 kg`, `12 m/s`, `b=2,2 m`, `β=18°`, `LOA=6,5 m`, `τ=3,0916°`, `L_K=6,051 m`. Os valores completos de `L_C`, `L_C2`, `T`, `lcp`, `lcg`, `vcg`, `Ixx`, `Izz` e `Ixz` daquela execução **não foram fornecidos nesta conversa**. Assim, não é cientificamente correto alegar que um exemplo genérico reproduz exatamente a abscissa original.

O `compare_legacy_and_fsd(boat)` foi criado precisamente para comparar o caso real, preservando os parâmetros do objeto. A diferença numérica da abscissa nessa comparação **não é**, por si, evidência de aumento ou diminuição de estabilidade física.

## 5. Cobertura e limitações

- 16 novos testes isolados da 28C.5.
- 74 testes prévios disponíveis nos pacotes 28C.1–28C.4 foram reexecutados: **90/90**, em ambiente local de desenvolvimento.
- Inversão das componentes transversais de força/momento no termo `h`, conversão de produto de inércia, distinção entre inversão isolada e rotação completa, similaridade da matriz de estado, autovalores, e polinômio normalizado, preservação do legado, versões rejeitadas e comparação não mutante.
- Não substitui nem afirma executar a suíte de 113 testes do fork; ela deverá ser repetida no ambiente do usuário, assim como o teste com o `PlaningBoat` real.
- Projeções por bordo de 1996 e dependência empírica do comprimento `L=(L_K+L_C)/2` permanecem hipóteses a validar contra fontes/ensaios.
- Termos de apêndices não incluídos; versão de 1997 dimensionalmente inconsistente continua bloqueada.
- A matriz resultante gera **diagnóstico matemático linear**, não aprovação normativa ou predição experimental de estabilidade.

## 6. Referências

- E. M. Lewandowski, *The Dynamics of Marine Craft: Maneuvering and Seakeeping*, 2004, cap. 1 e 6, eqs. 6.8–6.12.
- E. M. Lewandowski, *The Transverse Dynamic Stability of Planing Craft*, 1997, equações de movimento e apêndice de coeficientes.
- E. M. Lewandowski, 1996, formulação de forças normais nos dois bordos, equação 27.
- `docs/ETAPA_28C4_AUDITORIA_REFERENCIAIS.md` e `openplaning/stability/frame_audit.py` para convenções e transformações.
