# SemanticSeg
Segmentations
Пару раз в процессе работы приходилось сталкиваться с задачами сегментации, в репозиториях примеры решения данных задач (приходилось решать в колабе тк графика мака на куде не потянула бы). 

1) UNET - юнет для задачи сегментации, ничего особо нового не сделал, но пайплайн рабочий, просто бери и пользуйся. 

2) VGG2RF - штука чуть сложнее, делал по видосу в ютубе. По сути обычный трансфер весов последнего скрытого слоя предобученной VGG в случайный лес, работает чуть лучше нежели просто предобученная VGG, но и дольше тк работают последовательно две модели. 

3) Пробовал также маск RCNN и FasterRCNN, но тк код не менял особо, просто использовал собственный датасет смысла смотреть на такое думаю нет. 
