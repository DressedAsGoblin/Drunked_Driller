# Gtid Interpolation
Для запуска необходим Python версии 3.7+, пакеты:
- numpy
- matplotlib
- scipy
- tqdm

Перед запуском убедитесь, что все файлы гридов находятся в директории и проименованы в следующем порядке (n - количество пластов):
- Porosity_STD_[1-n]
- Saturation_STD_[1-n]
- Thickness_STD_[1-n]

Аргументы для запуска:
- `-c` или `--Case` - Указывает директорию, в которой находятся файлы.
- `-r` или `--Radius` - Радиус интерполяции.
- `-t` или `--Threshold` - Порог для свойств (в %).
- `-m` или `--Min` - Минимальное количество пластов в зоне интерполяции.
- `-n` или `--Normalize` - Включает log-нормализацию для гридов (если распределения свойств экспоненциальны).
- `--Std` - Коэффициент k для значения std по свойствам в %. Можно поставить 0 чтобы нивелировать влияние STD на результат.

Например, для работы с папкой Case_3, k=0, threshold=15 и min_plasts=2, аргументы выглядят следующим образом:
> python grid_interpolation.py -c "Case_3" --Std 0 -t 15 -m 2
