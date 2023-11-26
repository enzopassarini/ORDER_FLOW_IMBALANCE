# Order Flow Imbalance
Desenvolvimento de um possível sinal para HFT/Market Making baseado no conceito de Order Flow Imbalance (OFI).
<br><br>

Baseado no artigo "The price impact of order book events - Rama Cont, Arseniy Kukanov and Sasha Stoikov", esse indicador calcula um número a cada alteração no L1 do book (best bid/ask), considerando mudanças de preço e no volume naquele nível. Assim, um aumento no volume do bid, mesmo sem um aumento de preço, representaria mais desequilíbrio e mais força do lado comprador.
<br><br>
Nesse repositório foi criado o sistema central para a geração desse feature, podendo consequentemente ser usado para diversas aplicações a depender da tese formulada. Acumula-se o fator de OFI por um tempo pré determinado.
<br><br>
Um exemplo: order flow imbalance acumulado nos últimos 15 segundos demonstrou muito mais força compradora que vendedora, e minha pressuposição é de momentum (continuação do movimento). Isso geraria um sinal para uma ordem de compra.

