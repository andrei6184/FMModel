# FMModel

Проект представляет собой реализацию Factorization Machine на языке Scala.

## Архитектура приложения:

Утилита имеет два режима работы – обучение и применение. Исходные данные хранятся в текстовых файлах в CSV формате. Общая структура CSV файла следующая:
```
На этапе обучения: CF1, CF2, …, CFN, TARGET
На этапе применения: CF1, CF2, …, CFN
```
где **CFi** – соответствующий категориальный признак (целые числа);
**TARGET** – целевая переменная (вещественная);
Данные из файла считываются и преобразуются в **DataSet** коллекцию, которая содержит набор образцов в формате массива индексов соответсвующих категориальным признаком. Индекс из категориального признака формируется с помощью метода **HashingTrick**.
```
Index_i=CF_i%2^M
```
где *M* – это параметр, который задается через аргументы командной строки.
Следующий компонент системы это **Model** – компонет, который умеет предсказывать целевую переменную по входному образцу. Внутри данного компонента находятся веса модели.
Следующим компонентом системы является **SGD** –  компонент, который умеет обучать модель с помощью алгоритма стохастического градиента. На вход данному компоненту передается **Model** и **DataSet** а на выходе получается обученная модель.

## На данный момент реализовано:
	
1. парсинг CSV файлов;
2. хэширование признаков с помощью HashingTrick;
3. Factorization Machine модель которая состоит из трех параметров:
  3.1 bias – смещение;
  3.2 weights – обычный вектор весов скалярного произведения <wx>
  3.3 pairWeights – матрица парных весов участвующая в скалярном произведении<vj,vk>xjxk
4. сериализация, десериализация для модели в файл;
5. SGD – алгоритм, который обучает модель. Пока он работает только с задачей регрессии и использует MSE функцию потерь.
6. поддерживается работа с разреженными данными (матрицами и векторами);
7. используется алгоритм вычисления *a(x)* за время *O(n)*, где *n*-количество установленных параметров в векторе признаков образца.

## Что планируется реализовать:

1. SGD для задачи классификация (логистическую регрессию);
2. L1, L2 регуляризацию;
3. дружелюбный UI.


