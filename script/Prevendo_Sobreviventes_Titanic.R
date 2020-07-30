# ====================================
# Técnicas de Classificação
# ------------------------------------

#install.packages("tidyverse")
#install.packages("ggplot2")
#install.packages("titanic")

# ---------------------- Tratamento dos Dados --------------------------------------------------

library(tidyverse)
library(ggplot2)
library(titanic)

# Instalando pacote de HELP - usa a função findFn para saber detalhes de uma função
#install.packages("sos")
library(sos)

data("titanic_train")
data("titanic_test")

# ---------------------------- Preparando dataset de treino ----------------------------
str(titanic_train)
dados <- titanic_train
# Transformando em fator os atributos texto de interesse
dados <- dados %>% mutate(fSex = factor(Sex, levels = c('male','female')), 
                          fEmbarked = factor(Embarked, levels = c('S','C','Q')))

# Retirando o Sex(5) e Embarked(12) por estarem duplicadas
# A coluna PassengerId(1) porque nao tem significado
# Mais o ticket(9) e a cabine(11) por serem irrelevantes para a pesquisa
dados <- dados[,c(-1,-5,-9,-11,-12)]

# Verificando quais colunas existe conteudo N/A, se uma coluna tiver um N/A ela será excluída do dataset
v_na <- apply(dados, 2, is.na)
t_na <- apply(v_na, 2, sum)
t_na

# Como fEmbarked(9) soh possui dois elementos com N/A, optou-se por manter a coluna excluindo as linhas com problema
lin <- which(v_na[,9]) * -1
dados <- dados[lin,]

# Retirando Age por ter excesso de valor N/A
dados <- subset(dados, select = -Age)
treino <- dados

# Renomeando as colunas do data frame
colnames(treino) <- c('Sobreviveu','Classe','Nome','HFamilia','VFamilia','Preco','Sexo','Embarque')
str(treino)

# ----------------------------- Preparando dataset de teste -----------------------------
str(titanic_test)
dados <- titanic_test
# Transformando em fator os atributos texto de interesse
dados <- dados %>% mutate(fSex = factor(Sex, levels = c('male','female')), 
                          fEmbarked = factor(Embarked, levels = c('S','C','Q')))

# Retirando o Sex(4) e Embarked(11) por estarem duplicadas
# A coluna PassengerId(1) porque nao tem significado
# Mais o ticket(8) e a cabine(10) por serem irrelevantes para a pesquisa
dados <- dados[,c(-1,-4,-8,-10,-11)]

# Verificando quais colunas existe conteudo N/A, se uma coluna tiver um N/A ela será excluída do dataset
v_na <- apply(dados, 2, is.na)
t_na <- apply(v_na, 2, sum)
t_na

# Como Fare possui somente um elemento N/A, iremos retira-lo do dataset
lin <- which(v_na[,6]) * -1
dados <- dados[lin,]

# Retirando Age por ter excesso de valor N/A
dados <- subset(dados, select = -Age)
teste <- dados

# Renomeando as colunas do data frame
colnames(teste) <- c('Classe','Nome','HFamilia','VFamilia','Preco','Sexo','Embarque')
str(teste)

# Dividindo o vetor de treino
set.seed(548)
ind_treino <- sample(x = nrow(treino), size = 0.8*nrow(treino), replace = FALSE)
sample.int(n=5,size=4)

nrow(treino)
str(ind_treino)

valida <- treino[-ind_treino,]
treino <- treino[ind_treino,]
str(treino)
str(valida)

# Transformando a coluna Sobreviveu em fator para otimizar a aplicação nos modelos.
treino$Sobreviveu <- as.factor(treino$Sobreviveu)
valida$Sobreviveu <- as.factor(valida$Sobreviveu)
str(treino)
str(valida)


# ---------------------- Regressão Logistica --------------------------------------------------
str(treino)

# Ajustando considerando as variáveis Classe, Preço e Embarque somente, 
# pois são as variáveis que possuem alguma relação com critérios sócio-econômicos dos passageiros.
fit11 <- glm(Sobreviveu ~ Classe + Preco + Embarque + 1, data = treino, family = binomial())
summary(fit11)

# Retirou-se a variável Embarque devido ao não atendimento do nível de confiança.
fit12 <- glm(Sobreviveu ~ Classe + Preco + 1, data = treino, family = binomial())
summary(fit12)

# Foi mantido somente a variável Classe no modelo.
fit13 <- glm(Sobreviveu ~ Classe + 1, data = treino, family = binomial())
summary(fit13)

# O modelo escolhido foi Sobreviveu = Classe + Preco + Constante (segundo modelo), 
# pois o Critério de Informação de Akaike é menor que o terceiro modelo 
# e o primeiro modelo possui varíáveis que não atendem aos critérios estatísticos, 
# ou seja, ao nível de significância de 5% é provável que o coeficiente da variável relacionada ao Embarque seja nulo.
fit11$aic
fit12$aic
fit13$aic

# ROOT MEAN SQUARE ERROR (RMSE)
# A medida de erro mais comumente usada para aferir a qualidade do ajuste de
# um modelo é a chamada RAIZ DO ERRO MÉDIO QUADRÁTICO.
# Ela é a raiz do erro médio quadrático da diferença entre a predição e o valor real.
# Podemos pensar nela como sendo uma medida análoga ao desvio padrão.
library(ModelMetrics)

# De acordo com a Raiz do Erro Quadrático Médio, a escolha mais adequada é pelo segundo modelo, 
# apesar de não ter o menor RMSE, ele está muito próximo. 
# Isso está de acordo com a escolha do modelo pelo o AIC.
rmse(fit11)
rmse(fit12)
rmse(fit13)

# Modelo escolhido (segundo modelo)
summary(fit12)
fit12$coefficients
fit12$formula

# AIC e BIC do pacote stats
AIC(fit12)
BIC(fit12)

# Testando o modelo escolhido
str(valida)
previsao1 <- predict(fit12, newdata = valida, type = "response")
previsao1
str(previsao1)
previsao <- previsao1

valor_corte <- 0.5
# Criando a Matriz de Confusão manualmente
prev <- if_else(previsao > valor_corte, 1, 0)
prev <- as.factor(prev)
vp <- if_else(valida$Sobreviveu == 1 & prev == 1, 1, 0)  
fn <- if_else(valida$Sobreviveu == 1 & prev == 0, 1, 0)
fp <- if_else(valida$Sobreviveu == 0 & prev == 1, 1, 0)
vn <- if_else(valida$Sobreviveu == 0 & prev == 0, 1, 0)
m_conf <- rbind(cbind(sum(vn),sum(fn)),cbind(sum(fp),sum(vp)))
rownames(m_conf) <- c('Previsto Negativo','Previsto Positivo')
colnames(m_conf) <- c('Real Negativo','Real Positivo')
m_conf
m_conf[1,1]
m_conf[2,1]
m_conf[1,2]
m_conf[2,2]


real <- if_else(valida$Sobreviveu == 1, 1, 0)
# Utilizando a função do pacote modelMetrics para criar a Matriz de Confusão
m_conf2 <- confusionMatrix(real, previsao, cutoff = valor_corte)
rownames(m_conf2) <- c('Previsto Negativo','Previsto Positivo')
colnames(m_conf2) <- c('Real Negativo','Real Positivo')
m_conf2

# Precisão = VP / (VP + FP) - Indica o quanto um modelo gera de VP em relação a FP, ou seja, 
# quanto menor a Precisao mais ineficiente é um modelo, pois ele gera um alto FP que gera custos sem retorno.
precisao <- precision(real, previsao, cutoff = valor_corte)
precisao

# Precisão calculado manualmente
prc <- sum(vp) / (sum(vp) + sum(fp))
prc

# Sensibilidade (Recall) = VP / (VP + FN) - Indica o quanto o modelo gera de VP em comparação com FN, ou seja,
# mede o quanto o modelo é capaz de identificar todos os casos reais positivos.
sensibilidade <- recall(real, previsao, cutoff = valor_corte)
sensibilidade

# Recall calculado manualmente
rec <- sum(vp) / (sum(vp) + sum(fn))
rec

# Área sob a curva ROC - Mede a qualidade do modelo, quanto maior o valor de AUC, melhor o modelo.
auc(valida[,1],previsao)

# Obtendo o valor de corte otimizado para F1 Score máximo em relação ao Data Set de validação
previsto <- previsao
real <- if_else(valida$Sobreviveu == 1, 1, 0)

# >>> Definição da função f1_score <<<
f1_score <- function(limite) {
  x <- cbind(real,if_else(previsto > limite, 1, 0))
  colnames(x) = c('Real', 'Previsto')
  x <- as.data.frame(x)
  vp1 <- if_else(x$Real == 1 & x$Previsto == 1, 1, 0)
  fp1 <- if_else(x$Real == 0 & x$Previsto == 1, 1, 0)
  fn1 <- if_else(x$Real == 1 & x$Previsto == 0, 1, 0)
  r <- sum(vp1) / (sum(vp1) + sum(fn1))
  p <- sum(vp1) / (sum(vp1) + sum(fp1))
  2 * p * r / (p + r)
}

opt <- optimize(f1_score, c(0,1), tol = 0.0001, maximum = TRUE)
opt$maximum
opt$objective

# Medidas de Precisão e Sensibilidade com o valor de corte otimizado.
valor_corte <- opt$maximum
precisao <- precision(real, previsao, cutoff = valor_corte)
precisao
sensibilidade <- recall(real, previsao, cutoff = valor_corte)
sensibilidade
paste('F1 Score = ',f1Score(real, previsao, cutoff = valor_corte))
paste('Matriz de Confusão:')
m_conf2
  

# ---------------------- Árvore de Decisão --------------------------------------------------
# Pacote contendo as funções para trabalhar árvore de decisão
#install.packages("rpart")
#install.packages('rpart.plot')

library(rpart)
library(rpart.plot)

str(treino)

# Ajustando considerando as variáveis Classe, Preço e Embarque somente, 
# pois são as variáveis que possuem alguma relação com critérios sócio-econômicos dos passageiros.
fit21 <- rpart(Sobreviveu ~ Classe + Preco + Embarque + 1, data = treino)

# Exibição da árvore com plot - pacote padrão
nrow(treino)
par(fg = 'black', bg = 'cyan', lwd = 1.5, xpd = TRUE)
plot(fit21, compress = TRUE, margin = 0.15)
par(fg = 'red', bg = 'cyan', font = 2, xpd = TRUE)
text(fit21, all = FALSE, use.n = TRUE, fancy = FALSE)

# Identificando as paletas para usar nos gráficos do pacote rpart
show.prp.palettes()

# Exibição da árvore utilizando o pacote rpart.plot
par(fg = 'blue', bg = 'gray', font = 2, lwd = 2, xpd = TRUE)
rpart.plot(fit21, box.palette = c('OrPu'))


# Retirou-se a variável Embarque.
fit22 <- rpart(Sobreviveu ~ Classe + Preco + 1, data = treino)

# Exibição da árvore utilizando o pacote rpart.plot
par(fg = 'blue', bg = 'gray', font = 2, lwd = 2, xpd = TRUE)
rpart.plot(fit22, box.palette = c('OrPu'))


# Foi mantido somente a variável Classe no modelo.
fit23 <- rpart(Sobreviveu ~ Classe + 1, data = treino)

# Exibição da árvore utilizando o pacote rpart.plot
par(fg = 'blue', bg = 'gray', font = 2, lwd = 2, xpd = TRUE)
rpart.plot(fit23, box.palette = c('OrPu'))

# Como a classificação para ambos os modelos, primeiro e segundo, é basicamente a mesma, 
# optou-se pelo segundo modelo pois é mais parcimonioso.
summary(fit22)
par(fg = 'blue', bg = 'gray', font = 2, lwd = 2, xpd = TRUE)
rpart.plot(fit22, box.palette = c('OrPu'))

# Testando o modelo escolhido
str(valida)
previsao2 <- predict(fit22, newdata = valida)
str(previsao2)
head(previsao2)
head(previsao2[,2])
real <- if_else(valida$Sobreviveu == 1, 1, 0)
valor_corte <- 0.5
previsao <-previsao2[,2]
previsao

# Obtendo a precisão do modelo: Precisao = VP / (VP + FP), 
# que indica se o modelo é capaz de identificar os VP's errando pouco, ou seja, apontando falsos positivos.
precisao <- precision(real, previsao, valor_corte)
precisao

# Obtendo a sensibilidade do modelo: Sensibilidade = VP / (VP + FN),
# que indica se o modelo é capaz de identificar todos os VP's, deixando poucos sem identificar. 
sensibilidade <- recall(real, previsao, cutoff = valor_corte)
sensibilidade

# Obtendo o valor de corte otimizado para F1 Score máximo em relação ao Data Set de validação
previsto <- previsao
real <- if_else(valida$Sobreviveu == 1, 1, 0)

# >>> Usando a função f1_score definida acima
opt <- optimize(f1_score, c(0,1), tol = 0.0001, maximum = TRUE)
opt$maximum
opt$objective

# Medidas de Precisão e Sensibilidade com o valor de corte otimizado
valor_corte <- opt$maximum
valor_corte <- 0.32
precisao <- precision(real, previsao, cutoff = valor_corte)
precisao
sensibilidade <- recall(real, previsao, cutoff = valor_corte)
sensibilidade
paste("F1 Score = ", f1Score(real, previsao, cutoff = valor_corte))

real <- if_else(valida$Sobreviveu == 1, 1, 0)
# Utilizando a função do pacote modelMetrics para criar a Matriz de Confusão
m_conf2 <- confusionMatrix(real, previsao, cutoff = valor_corte)
rownames(m_conf2) <- c('Previsto Negativo','Previsto Positivo')
colnames(m_conf2) <- c('Real Negativo','Real Positivo')
paste('Matriz de Confusão:')
m_conf2


  # ---------------------- Análise de Discriminante --------------------------------------------------
# Pacote contendo funções para trabalhar Análise de Discriminante  
#install.packages("MASS")
library(MASS)

# Pacote que possui o teste Box-M para verificar homogeneidade de variância 
# e cálculo da distâcia de Mahalanobis
#install.packages("heplots")
library(heplots)

# Pacote que possui o teste Lambda de Wilks
#install.packages("rrcov")
library(rrcov)

str(treino)

# Ajustando considerando as variáveis Classe, Preço e Embarque somente, 
# pois são as variáveis que possuem alguma relação com critérios sócio-econômicos dos passageiros. 
fit31 <- lda(Sobreviveu ~ Classe + Preco + Embarque + 1, treino)
fit31
fit31$counts
fit31$svd
fit31$N

# Teste Lambda de Wilks, verifica se o modelo consegue classificar bem os grupo. 
# Quanto menor seu valor, melhor será a eficiência na discriminação. 
treino_exp <- treino %>% mutate(Num_Embarque = if_else(treino$Embarque == 'S', 1, if_else(treino$Embarque == 'C', 2, 3)))
str(treino_exp)
# Usando a formula
Wilks.test(Sobreviveu ~ Classe + Preco + Num_Embarque + 1, data = treino_exp)
# Usando os subset's do Data Set de treino
# *** O RESULTADO GERADO POR ESSA FUNÇÃO PRECISA SER VALIDADO ***
t_wilks <- Wilks.test(treino_exp[,c(2,6,9)], treino_exp[,1])
t_wilks
t_wilks$p.value
t_wilks$statistic
t_wilks$parameter

# Teste de Box M para verificar a igualdade das matrizes de variância e covariância entre os grupos.
# H_0: As matrizes de variância e covariância homogêneas (iguais).
# *** O RESULTADO GERADO POR ESSA FUNÇÃO PRECISA SER VALIDADO ***
str(treino_exp)
t_BoxM <- boxM(treino_exp[,c(2,6,9)], treino_exp[,1])
t_BoxM
t_BoxM$p.value
t_BoxM$statistic
t_BoxM$df
t_BoxM$cov
t_BoxM$means
t_BoxM$pooled

# Distância de Mahalanobis
# *** O RESULTADO GERADO POR ESSA FUNÇÃO PRECISA SER VALIDADO ***
t_mahalanobis <- Mahalanobis(treino_exp[,c(2,6,9)])
summary(t_mahalanobis)


# Retirou-se a variável Embarque, pois o primeiro modelo não atende a premissa da técnica
# que declara que as matrizes de variância e covariância devem ser homogêneas.
fit32 <- lda(Sobreviveu ~ Classe + Preco + 1, data = treino)
fit32
fit32$counts
fit32$svd
fit32$N

treino_exp <- treino
str(treino_exp)

# *** O RESULTADO GERADO POR ESSA FUNÇÃO PRECISA SER VALIDADO ***
t_wilks <- Wilks.test(treino_exp[,c(2,6)], treino_exp[,1])
t_wilks
t_wilks$p.value
t_wilks$statistic
t_BoxM <- boxM(treino_exp[,c(2,6)], treino_exp[,1])
t_BoxM
t_BoxM$p.value
t_BoxM$statistic

# Foi mantido somente a variável Classe no modelo.
fit33 <- lda(Sobreviveu ~ Classe + 1, data = treino)
fit33
fit33$counts
fit33$svd
fit33$N

treino_exp <- valida
str(treino_exp)

# *** O RESULTADO GERADO POR ESSA FUNÇÃO PRECISA SER VALIDADO ***
t_wilks <- Wilks.test(treino_exp[,2], treino_exp[,1])
t_wilks
t_wilks$p.value
t_wilks$statistic
t_BoxM <- boxM(treino_exp[,2], treino_exp[,1])
t_BoxM
t_BoxM$p.value
t_BoxM$statistic

# Como a classificação para ambos os modelos, primeiro, segundo e terceiro, é basicamente a mesma, 
# optou-se pelo segundo modelo pois é mais parcimonioso e possui um desvio padrão melhor que o primeiro
# e não é tão simples quanto o terceiro.
# Uma questão importante a mencionar foi que nenhum dos modelos passou no teste BoxM, 
# ou seja, não há garantias que as matrizes de variância das classes sejam semelhantes.
# Apesar disso, mantemos a análise através da técnica.
fit32
fit32$counts
fit32$svd
fit32$N

# Testando o modelo escolhido
str(valida)
previsao3 <- predict(fit2, newdata = valida)
str(previsao3)
ls(previsao3)
previsao3$class
head(previsao3$posterior,10)
head(previsao3$x,10)
real <- if_else(valida$Sobreviveu == 1, 1, 0)
valor_corte <- 0.5
previsao <- previsao3

# Obtendo a precisão do modelo: Precisao = VP / (VP + FP), 
# que indica se o modelo é capaz de identificar os VP's errando pouco, ou seja, apontando falsos positivos.
precisao <- ModelMetrics::precision(real, previsao$posterior[,2], valor_corte)
precisao

# Obtendo a sensibilidade do modelo: Sensibilidade = VP / (VP + FN),
# que indica se o modelo é capaz de identificar todos os VP's, deixando poucos sem identificar. 
sensibilidade <-  ModelMetrics::recall(real, previsao$posterior[,2], cutoff = valor_corte)
sensibilidade

# Obtendo o valor de corte otimizado para F1 Score máximo em relação ao Data Set de validação
previsto <- previsao$posterior[,2]
real <- if_else(valida$Sobreviveu == 1, 1, 0)

# >>> Usando a função f1_score definida acima
opt <- optimize(f1_score, c(0,1), tol = 0.0001, maximum = TRUE)
opt$maximum
opt$objective

# Medidas de Precisão e Sensibilidade com o valor de corte otimizado
valor_corte <- opt$maximum
precisao <- ModelMetrics::precision(real, previsao$posterior[,2], cutoff = valor_corte)
precisao
sensibilidade <-  ModelMetrics::recall(real, previsao$posterior[,2], cutoff = valor_corte)
sensibilidade
paste("F1 Score = ",  ModelMetrics::f1Score(real, previsao$posterior[,2], cutoff = valor_corte))

real <- if_else(valida$Sobreviveu == 1, 1, 0)
# Utilizando a função do pacote modelMetrics para criar a Matriz de Confusão
m_conf2 <-  ModelMetrics::confusionMatrix(real, previsao$posterior[,2], cutoff = valor_corte)
rownames(m_conf2) <- c('Previsto Negativo','Previsto Positivo')
colnames(m_conf2) <- c('Real Negativo','Real Positivo')
paste('Matriz de Confusão:')
m_conf2


# ---------------------- k-NN (K - Nearest Neighbors) --------------------------------------------------
# Pacote contendo as funções para trata k-NN
#install.packages('class')
library(class)

help(knn)

# Setando o k para todos os ajustes
k_vizinho <- 30

# [ ::: Primeiro Ajuste ::: ]
# Ajustando considerando as variáveis Classe, Preço e Embarque somente, 
# pois são as variáveis que possuem alguma relação com critérios sócio-econômicos dos passageiros. 
str(treino)
matriz_casos <- treino[,c(2,6)]
matriz_casos <- matriz_casos %>% 
  mutate(Num_Embarque = if_else(treino$Embarque == 'S', 1, if_else(treino$Embarque == 'C', 2, 3)))
classe_casos <- treino[,1]

str(valida)
matriz_teste <- valida[,c(2,6)]
matriz_teste <- matriz_teste %>% 
  mutate(Num_Embarque = if_else(valida$Embarque == 'S', 1, if_else(valida$Embarque == 'C', 2, 3)))

str(matriz_casos)
str(matriz_teste)
str(classe_casos)

fit41 <- knn(matriz_casos, matriz_teste, classe_casos, k = k_vizinho, prob = TRUE)

str(fit41)
ls(attributes(fit41))
str(attributes(fit41))

# Criando a Matriz de Confusão manualmente
prev <- fit41
prev <- as.factor(prev)
vp <- if_else(valida$Sobreviveu == 1 & prev == 1, 1, 0)  
fn <- if_else(valida$Sobreviveu == 1 & prev == 0, 1, 0)
fp <- if_else(valida$Sobreviveu == 0 & prev == 1, 1, 0)
vn <- if_else(valida$Sobreviveu == 0 & prev == 0, 1, 0)
m_conf <- rbind(cbind(sum(vn),sum(fn)),cbind(sum(fp),sum(vp)))
rownames(m_conf) <- c('Previsto Negativo','Previsto Positivo')
colnames(m_conf) <- c('Real Negativo','Real Positivo')
matriz_confusao_1 <- m_conf

# Obtendo a precisão do modelo: Precisao = VP / (VP + FP), 
# que indica se o modelo é capaz de identificar os VP's errando pouco, ou seja, apontando falsos positivos.
prc1 <- sum(vp) / (sum(vp) + sum(fp))

# Obtendo a sensibilidade do modelo: Sensibilidade = VP / (VP + FN),
# que indica se o modelo é capaz de identificar todos os VP's, deixando poucos sem identificar. 
rec1 <- sum(vp) / (sum(vp) + sum(fn))

# Obtendo a medida F1 Score
f1score1 <- 2 * prc1 * rec1 / (prc1 + rec1)


# [ ::: Segundo Ajuste ::: ]
# Retirou-se a variável Embarque.
str(treino)
matriz_casos <- treino[,c(2,6)]
classe_casos <- treino[,1]

str(valida)
matriz_teste <- valida[,c(2,6)]

str(matriz_casos)
str(matriz_teste)
str(classe_casos)

fit42 <- knn(matriz_casos, matriz_teste, classe_casos, k = k_vizinho, prob = TRUE)

str(fit42)
ls(attributes(fit42))
str(attributes(fit42))

# Criando a Matriz de Confusão manualmente
prev <- fit42
prev <- as.factor(prev)
vp <- if_else(valida$Sobreviveu == 1 & prev == 1, 1, 0)  
fn <- if_else(valida$Sobreviveu == 1 & prev == 0, 1, 0)
fp <- if_else(valida$Sobreviveu == 0 & prev == 1, 1, 0)
vn <- if_else(valida$Sobreviveu == 0 & prev == 0, 1, 0)
m_conf <- rbind(cbind(sum(vn),sum(fn)),cbind(sum(fp),sum(vp)))
rownames(m_conf) <- c('Previsto Negativo','Previsto Positivo')
colnames(m_conf) <- c('Real Negativo','Real Positivo')
matriz_confusao_2 <- m_conf

# Obtendo a precisão do modelo: Precisao = VP / (VP + FP), 
# que indica se o modelo é capaz de identificar os VP's errando pouco, ou seja, apontando falsos positivos.
prc2 <- sum(vp) / (sum(vp) + sum(fp))

# Obtendo a sensibilidade do modelo: Sensibilidade = VP / (VP + FN),
# que indica se o modelo é capaz de identificar todos os VP's, deixando poucos sem identificar. 
rec2 <- sum(vp) / (sum(vp) + sum(fn))

# Obtendo a medida F1 Score
f1score2 <- 2 * prc2 * rec2 / (prc2 + rec2)


# [ ::: Terceiro Ajuste ::: ]
# Foi mantido somente a variável Classe no modelo.
str(treino) 
matriz_casos <- treino[,2]
matriz_casos <- as.data.frame(matriz_casos)
classe_casos <- treino[,1]

str(valida)
matriz_teste <- valida[,2]
matriz_teste <- as.data.frame(matriz_teste)

str(matriz_casos)
str(matriz_teste)
str(classe_casos)

fit43 <- knn(matriz_casos, matriz_teste, classe_casos, k = k_vizinho, prob = TRUE)

str(fit43)
ls(attributes(fit43))
str(attributes(fit43))

# Criando a Matriz de Confusão manualmente
prev <- fit43
prev <- as.factor(prev)
vp <- if_else(valida$Sobreviveu == 1 & prev == 1, 1, 0)  
fn <- if_else(valida$Sobreviveu == 1 & prev == 0, 1, 0)
fp <- if_else(valida$Sobreviveu == 0 & prev == 1, 1, 0)
vn <- if_else(valida$Sobreviveu == 0 & prev == 0, 1, 0)
m_conf <- rbind(cbind(sum(vn),sum(fn)),cbind(sum(fp),sum(vp)))
rownames(m_conf) <- c('Previsto Negativo','Previsto Positivo')
colnames(m_conf) <- c('Real Negativo','Real Positivo')
matriz_confusao_3 <- m_conf

# Obtendo a precisão do modelo: Precisao = VP / (VP + FP), 
# que indica se o modelo é capaz de identificar os VP's errando pouco, ou seja, apontando falsos positivos.
prc3 <- sum(vp) / (sum(vp) + sum(fp))

# Obtendo a sensibilidade do modelo: Sensibilidade = VP / (VP + FN),
# que indica se o modelo é capaz de identificar todos os VP's, deixando poucos sem identificar. 
rec3 <- sum(vp) / (sum(vp) + sum(fn))

# Obtendo a medida F1 Score
f1score3 <- 2 * prc3 * rec3 / (prc3 + rec3)


#Comparação do resultado para os três ajustes. 
# Há ligeira mudança na qualidade com a retirada das variáveis, 
# sendo percepitível com a mudança do valor de k.
# O modelo que se mostrou melhor foi o modelo 1 com k igual a 30.
paste("Primeiro Ajuste: ", "Precisão=", prc1, " - Sensibilidade=", rec1, " - F1 Score=", f1score1)
paste("Segundo Ajuste: ", "Precisão=", prc2, " - Sensibilidade=", rec2, " - F1 Score=", f1score2)
paste("Terceiro Ajuste: ", "Precisão=", prc3, " - Sensibilidade=", rec3, " - F1 Score=", f1score3)
matriz_confusao_1
matriz_confusao_2
matriz_confusao_3


# ---------------------- Random Forest --------------------------------------------------
# Obtendo o pacote com as funções de RandomForest
#install.packages("randomForest")
library(randomForest)

# Obtendo pacote com funções utilitárias de análise de classificação
#install.packages("caret")
library(caret)


# Ajustando o Data Set para todos os valores serem númericos
# para funcionar melhor com as funções do pacote
str(treino)
treino_exp <- treino %>% 
  mutate(Num_Embarque = if_else(treino$Embarque == 'S', 1, if_else(treino$Embarque == 'C', 2, 3)))
treino_exp <- treino_exp[,c(-3,-4,-5,-7,-8)]
colnames(treino_exp) <- c('Sobreviveu','Classe','Preco','Embarque')
str(treino_exp)


# Ajustando considerando as variáveis Classe, Preço e Embarque somente, 
# pois são as variáveis que possuem alguma relação com critérios sócio-econômicos dos passageiros.
fit51 <- randomForest(Sobreviveu ~ Classe + Preco + Embarque + 1, data = treino_exp, proximity = TRUE)
fit51
ls(fit51)
fit51$predicted
fit51$y
fit51$votes
fit51$confusion
fit51$ntree
fit51$proximity
tail(fit51$err.rate)

par(fg = 'blue', bg = 'gray', lwd = 1.5, xpd = TRUE)
plot(fit51, main = 'Random Forest - Primeiro Ajuste')

par(fg = 'yellow', bg = 'gray', lwd = 2, xpd = TRUE)
partialPlot(fit51, pred.data = treino_exp, x.var = Classe, which.class = '1')
partialPlot(fit51, pred.data = treino_exp, x.var = Preco, which.class = '1')
partialPlot(fit51, pred.data = treino_exp, x.var = Embarque, which.class = '1')

par(fg = 'black', bg = 'gray', lwd = 1, xpd = TRUE)
MDSplot(fit51, k = 2, fac = treino_exp$Sobreviveu, palette=c('red', 'blue'), pch=20)


# Retirou-se a variável Embarque.
fit52 <- randomForest(Sobreviveu ~ Classe + Preco + 1, data = treino_exp, proximity = TRUE)
fit52

par(fg = 'blue', bg = 'gray', lwd = 1.5, xpd = TRUE)
plot(fit52, main = 'Random Forest - Segundo Ajuste')

par(fg = 'yellow', bg = 'gray', lwd = 2, xpd = TRUE)
partialPlot(fit52, pred.data = treino_exp, x.var = Classe, which.class = '1')
partialPlot(fit52, pred.data = treino_exp, x.var = Preco, which.class = '1')
partialPlot(fit52, pred.data = treino_exp, x.var = Embarque, which.class = '1')

par(fg = 'black', bg = 'gray', lwd = 1, xpd = TRUE)
MDSplot(fit52, k = 2, fac = treino_exp$Sobreviveu, palette=c('red', 'blue'), pch=20)


# Foi mantido somente a variável Classe no modelo.
fit53 <- randomForest(Sobreviveu ~ Classe + 1, data = treino_exp, proximity = TRUE)
fit53

par(fg = 'blue', bg = 'gray', lwd = 1.5, xpd = TRUE)
plot(fit53, main = 'Random Forest - Terceiro Ajuste')

par(fg = 'yellow', bg = 'gray', lwd = 2, xpd = TRUE)
partialPlot(fit53, pred.data = treino_exp, x.var = Classe, which.class = '1')
partialPlot(fit53, pred.data = treino_exp, x.var = Preco, which.class = '1')
partialPlot(fit53, pred.data = treino_exp, x.var = Embarque, which.class = '1')

par(fg = 'black', bg = 'gray', lwd = 1, xpd = TRUE)
MDSplot(fit53, k = 2, fac = treino_exp$Sobreviveu, palette=c('red', 'blue'), pch=20)

# De acordo com uma avaliação na Precisao e Recall dos três ajustes, 
# optou-se pelo Terceiro Ajuste pois produziu um F1 Score melhor.
fit51$confusion
class(fit51$confusion)
m <- fit51$confusion
# VP / (VP + FP)
precisao <- m[2,2] / (m[2,2] + m[2,1])
precisao
# VP / (VP + FN)
recall <- m[2,2] / (m[2,2] + m[1,2])
recall
f1score <- 2 * precisao * recall / (precisao + recall)
f1score

fit52$confusion
m <- fit52$confusion
# VP / (VP + FP)
precisao <- m[2,2] / (m[2,2] + m[2,1])
precisao
# VP / (VP + FN)
recall <- m[2,2] / (m[2,2] + m[1,2])
recall
f1score <- 2 * precisao * recall / (precisao + recall)
f1score

fit53$confusion
m <- fit53$confusion
# VP / (VP + FP)
precisao <- m[2,2] / (m[2,2] + m[2,1])
precisao
# VP / (VP + FN)
recall <- m[2,2] / (m[2,2] + m[1,2])
recall
f1score <- 2 * precisao * recall / (precisao + recall)
f1score


# Testando o modelo escolhido - Segundo modelo
str(valida)
previsao <- predict(fit52, newdata = valida)
str(previsao)
previsao
prev <- as.factor(previsao)
length(prev)
length(valida$Sobreviveu)

# Obtendo a matriz de confusão e demais métricas de validação do pacote caret
m_conf <- caret::confusionMatrix(data = prev, reference = valida$Sobreviveu, positive = '1', mode = "everything")
m_conf

ls(m_conf)
m_conf$byClass
m_conf$overall
m_conf$table

# Precisão calculado através do pacote caret
precisao <- caret::precision(data = prev, reference = valida$Sobreviveu, relevant = '1')
precisao

# Recall calculado através do pacote caret
recall <- caret::recall(data = prev, reference = valida$Sobreviveu, relevant = '1')
recall  

# Limitando o número de árvores igual a 100
fit54 <- randomForest(Sobreviveu ~ Classe + Preco + 1, data = treino_exp, proximity = TRUE, ntree = 100)
fit54
par(fg = 'blue', bg = 'gray', lwd = 1.5, xpd = TRUE)
plot(fit54, main = 'Random Forest - Ajuste Extra')

fit54$confusion
m <- fit54$confusion
# VP / (VP + FP)
precisao <- m[2,2] / (m[2,2] + m[2,1])
precisao
# VP / (VP + FN)
recall <- m[2,2] / (m[2,2] + m[1,2])
recall
f1score <- 2 * precisao * recall / (precisao + recall)
f1score
