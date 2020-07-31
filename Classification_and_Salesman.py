import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def euclid(a, b):
	"""
	Если можно провести прямую от a до b
	:param tuple a: x, y of point a
	:param tuple b: x, y of point b
	:return: Euclid distance
	"""
	return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


def check_distances(markers, id_of_cores):
	for marker in markers:
		id_of_closer_core = 0
		minimal_distance = size*2.0
		for core_id in id_of_cores:
			if marker.id != core_id:  # не сравнивается сам с собой
				new_distance = euclid((marker.x, marker.y), (markers[core_id].x, markers[core_id].y))
				if new_distance < minimal_distance:
					minimal_distance = new_distance
					id_of_closer_core = markers[core_id].id
		if not marker.klass_core:
			marker.klass_id = markers[id_of_closer_core].klass_id
			marker.color = markers[id_of_closer_core].color


def correcting():
	global changed
	changed = False
	id_of_cores = [elem.id for elem in markers if elem.klass_core]
	for core_id in id_of_cores:
		sum_of_x = 0
		sum_of_y = 0
		i = 0
		for marker in markers:
			if marker.id != markers[core_id].id:
				if marker.klass_id == markers[core_id].klass_id:
					sum_of_x += marker.x
					sum_of_y += marker.y
					i += 1
		if i:
			new_x = sum_of_x / i
			new_y = sum_of_y / i
			if new_x != markers[core_id].x:
				markers[core_id].x = new_x
				changed = True
			if new_y != markers[core_id].y:
				markers[core_id].y = new_y
				changed = True


class Marker:
	core_counter = 0
	id_counter = 0
	COLORS = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#ff7f0e', '#17becf']

	def __init__(self, core=None):
		self.id = Marker.id_counter
		Marker.id_counter += 1
		if core:
			try:
				self.color = Marker.COLORS[self.id]
			except IndexError:
				self.color = '#%02X%02X%02X' % (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
		else:
			self.color = Marker.COLORS[0]
		self.x = random.randint(0, size)
		self.y = random.randint(0, size)
		if core:
			self.klass_id = Marker.core_counter
			self.klass_core = True
			Marker.core_counter += 1
		else:
			self.klass_id = 0
			self.klass_core = False

	def get_x_y(self):
		return self.x, self.y

	@staticmethod
	def restart():
		Marker.core_counter = 0
		Marker.id_counter = 0


class Salesman:
	def __init__(self, mapsize, numberofstops, maxmen, mutationrate, t, a):
		self.numberofstops = numberofstops  # количество остановок
		self.mutationrate = mutationrate  # определяет вероятность мутации
		self.targets = np.random.randint(mapsize, size=(self.numberofstops, 2))  # определяем местоположение городов по х, у
		self.men = np.empty((maxmen, self.numberofstops), dtype=np.int32)  # создаём пустой массив для маршрутов
		for number in range(maxmen):
			tempman = np.arange(self.numberofstops, dtype=np.int32)  # создаём последовательность индексов городов от 0 до кол-ва_городов
			np.random.shuffle(tempman)
			self.men[number] = tempman
		self.best = None
		self.bestlength = 0

		# МОНТЕ-КАРЛО
		self.route = np.arange(self.numberofstops, dtype=np.int32)
		self.distance = self.determine_distance(self.route)
		self.t = t
		self.a = a

	def calculate_lengths(self):
		# создаём пустой массив (индекс маршрута, длина)
		temporder = np.empty([len(self.men), 2], dtype=np.int32)
		# записываем индексы маршрутов в temporder до того как будем менять их
		for number in range(len(self.men)):
			temporder[number] = [number, 0]  # [индекс маршрута(self.men), длина маршрута]
		# вычисляем длину пути для всех маршрутов (коммивояжёров)
		for number in range(len(self.men)):
			templength = 0
			# для всех городов
			for target in range(len(self.targets) - 1):
				templength += round(abs(euclid(self.targets[self.men[number][target]], self.targets[self.men[number][target + 1]])), 2)
			# считаем длину возвращения в исходный город (старт)
			templength += round(abs(euclid(self.targets[self.men[number][0]], self.targets[self.men[number][-1]])), 2)
			temporder[number][1] = templength
		# сортируем маршруты по длине пути для вывода
		temporder = sorted(temporder, key=lambda x: -x[1])
		# возвращаем вторую половину (лучшую) маршрутов (коммивояжёров)
		return temporder

	def breed(self, parent1_id, parent2_id):
		cut_id = random.randint(1, len(self.men[parent1_id])-1)  # -1, чтобы не было такого, что один из массивов пустой
		dna1 = np.copy(self.men[parent1_id][:cut_id])  # копия, чтобы не изменялись оригиналы
		dna2 = np.copy(self.men[parent2_id][cut_id:])
		#print("MEN", self.men)
		#print(self.men[parent1_id][:cut_id], self.men[parent2_id][cut_id:])

		no_cities = []  # потеряные города в результате разбиения
		for y in range(self.numberofstops):
			if y not in dna1 and y not in dna2:
					no_cities.append(y)

		for index in range(len(dna2)):
			if not no_cities:
				break
			if dna2[index] in dna1:
				dna2[index] = random.choice(no_cities)  # для этого индекса случайно выбираем город из no_cities и добавляем его в dna2
				no_cities.remove(dna2[index])  # и удаляем из no_cities
		# скрещенный потомок
		offspring1 = np.append(dna1, dna2)
		while no_cities:
			offspring1 = np.append(offspring1, no_cities[0])
			no_cities.pop(0)

		# мутация
		if random.random() <= self.mutationrate:
			j = random.randint(0, self.numberofstops - 1)
			# меняем два гена местами
			g = random.randint(0, self.numberofstops - 1)
			while j == g:
				g = random.randint(0, self.numberofstops - 1)
			temp = offspring1[j]
			offspring1[j] = offspring1[g]
			offspring1[g] = temp

		# берём остальные части хромосом родителей
		dna1 = np.copy(self.men[parent2_id][:cut_id])
		dna2 = np.copy(self.men[parent1_id][cut_id:])

		no_cities = []  # потеряные города в результате разбиения
		for y in range(self.numberofstops):
			if y not in dna1 and y not in dna2:
				no_cities.append(y)

		for index in range(len(dna2)):
			if not no_cities:
				break
			if dna2[index] in dna1:
				dna2[index] = random.choice(no_cities)  # для этого индекса случайно выбираем город из no_cities и добавляем его в dna2
				no_cities.remove(dna2[index])  # и удаляем из no_cities
		# скрещенный потомок
		offspring2 = np.append(dna1, dna2)
		while no_cities:
			offspring2 = np.append(offspring2, no_cities[0])
			no_cities.pop(0)

		# мутация
		if random.random() <= self.mutationrate:
			j = random.randint(0, self.numberofstops - 1)
			g = random.randint(0, self.numberofstops - 1)
			while j == g:
				g = random.randint(0, self.numberofstops - 1)
			temp = offspring2[j]
			offspring2[j] = offspring2[g]
			offspring2[g] = temp

		# выживание
		survivors = (self.men[parent1_id], self.men[parent2_id], offspring1, offspring2)
		lengths = []
		# вычисляем длину пути для всех маршрутов (коммивояжёров)
		for surv in survivors:
			templength = 0
			for target in range(len(self.targets) - 1):
				templength += round(abs(euclid(self.targets[surv[target]], self.targets[surv[target + 1]])), 2)
			# считаем длину возвращения в исходный город (старт)
			templength += round(abs(euclid(self.targets[surv[0]], self.targets[surv[-1]])), 2)
			lengths.append(templength)
		short_id = lengths.index(min(lengths))
		return survivors[short_id]

	def breed_new_generation(self):
		have_partner = []
		for i in range(len(self.men)-1):
			if i in have_partner:
				continue
			parent1_id = i
			parent2_id = 0
			while parent2_id in have_partner or parent1_id == parent2_id:
				parent2_id = random.randint(i+1, len(self.men)-1)
			have_partner.append(parent1_id)
			have_partner.append(parent2_id)
			self.men[i] = self.breed(parent1_id, parent2_id)

	# МОНТЕ-КАРЛО
	def determine_distance(self, route):
		new_distance = 0
		for i in range(len(self.targets) - 1):
			new_distance += round(abs(euclid(self.targets[route[i]], self.targets[route[i + 1]])), 2)
		# считаем длину возвращения в исходный город (старт)
		new_distance += round(abs(euclid(self.targets[route[0]], self.targets[route[-1]])), 2)
		return new_distance

	def get_new_route(self):
		i = np.random.randint(0, len(self.route))
		j = np.random.randint(0, len(self.route))
		while i == j:
			j = np.random.randint(0, len(self.route))
		buff = self.route[i]
		new_route = np.copy(self.route)
		new_route[i] = self.route[j]
		new_route[j] = buff

		new_distance = self.determine_distance(new_route)
		return new_route, new_distance

	def calculate(self, animator):
		global len_sum, iteration, i, total_i, accuracy, iterations, asked

		if len_sum < accuracy:
			print("\nГА")
			self.breed_new_generation()
			lens = self.calculate_lengths()
			self.best, self.bestlength = np.array(lens)[-1]
			len_sum = abs(self.bestlength - np.array(lens)[0][1])  # Разница длин лучшего и наихудшего коммивояжёров
			print("На итерации {} минимальное расчётное расстояние составило {}.".format(iteration, self.bestlength))
			print("Разница лучшей и худшей длин = {}, предел = {}".format(len_sum, accuracy))
			iteration += 1

		if i < iterations:
			print("\nМОНТЕ-КАРЛО")
			self.distance = self.determine_distance(self.route)
			new_route, new_distance = self.get_new_route()
			diff = self.distance - new_distance
			if diff > 0 or diff < 0 and np.random.random() <= np.exp(diff / self.t):
				self.route = new_route
				self.distance = round(new_distance, 2)
				i = 0
			else:
				i += 1
			self.t *= self.a
			#if self.t < 0.001:
			#	self.t = 0.001
			print("Total_i", "i", "Расстояние", "Температура")
			print(total_i, i, self.distance, self.t)
			total_i += 1

		if len_sum >= accuracy and iterations == i:
			print("\nМинимальное расчётное расстояние (ГА):", self.bestlength)
			print("Минимальное расчётное расстояние (Монте-Карло):", self.distance)
			i += 1
		#ГА
		ax1.clear()
		ax1.title.set_text("Генетический алгоритм")
		ax1.scatter(self.targets[[..., 0]], self.targets[[..., 1]], s=20)
		linearray = self.targets[self.men[self.best]]
		linearray = np.append(linearray, [linearray[0]], axis=0)
		ax1.plot(linearray[[..., 0]], linearray[[..., 1]])
		#МК
		ax2.clear()
		ax2.title.set_text("Метрополис")
		ax2.scatter(self.targets[[..., 0]], self.targets[[..., 1]], s=20)
		linearray = self.targets[self.route]
		linearray = np.append(linearray, [linearray[0]], axis=0)
		ax2.plot(linearray[[..., 0]], linearray[[..., 1]])


work = True
while work:
	selector = ''
	while selector not in ('1', '2', '0'):
		print("\nМЕНЮ\n1) Классификация (ISODATA и СК)\n2) Решение задачи коммивояжёра (ГА и МК)\n0) Выход\n")
		selector = input()
	if selector == '0':
		work = False
	elif selector == '1':
		changed = True
		iteration = 1
		Marker.core_counter = 0
		Marker.id_counter = 0
		m = 0  # markers
		c = 0  # classes
		size = 100

		while m < 3:
			print("Введите число точек, которое должно быть больше 2: ")
			try:
				m = int(input())
			except ValueError:
				print("Введите целочисленное значения: ")

		while m <= c or c < 2:
			print("Введите число классов, которое должно быть меньше числа точек, но больше 1: ")
			try:
				c = int(input())
			except ValueError:
				print("Введите целочисленное значения: ")

		# ISODATA
		m += c
		beginning = True
		while beginning:
			markers = []  # (x, y, class_id)
			id_of_cores = []
			for i in range(m):
				if i < c:
					markers.append(Marker(core=True))
					id_of_cores.append(markers[len(markers) - 1].id)
				else:
					markers.append(Marker())
			# разбросали маркеры

			cores_data = [[], [], []]  # all x, all y, all colors
			points_data = [[], [], []]
			for elem in markers:
				if elem.klass_core:
					cores_data[0].append(elem.x)
					cores_data[1].append(elem.y)
					cores_data[2].append(elem.color)
				else:
					points_data[0].append(elem.x)
					points_data[1].append(elem.y)
					points_data[2].append(elem.color)

			plt.scatter(cores_data[0], cores_data[1], c=cores_data[2], alpha=0.8, marker='X')  # [elem.x for elem in markers if not elem.core_head], [elem.y for elem in markers if not elem.core_head], 'ro'
			plt.scatter(points_data[0], points_data[1], c='black', alpha=0.8, marker='.')
			plt.axis([0, size + 1, 0, size + 1])
			print("Вас устраивает расположение ядер?")
			plt.show()
			print("Введите 0, если не устраивает. Нажмите Enter.")
			if input() != "0":
				beginning = False
			else:
				Marker.restart()

		# Kohonen
		XY = [marker.get_x_y() for marker in markers[c:]]  # [c:] чтобы не брать в расчёт классы из ISODATA
		weights = list()  # список для весов
		edu_rate = 0.0#0.3  # коэффициент обучения
		edu_deteriorate = 0.0#0.05  # уменьшение коэффициента обучения
		while edu_rate <= 0.0:
			print("Задайте коэффициент обучения:")
			try:
				edu_rate = float(input())
			except ValueError:
				print("Введите корректное значение!")
		while edu_deteriorate <= 0.0:
			print("Задайте уменьшение коэффициента обучения:")
			try:
				edu_deteriorate = float(input())
			except ValueError:
				print("Введите корректное значение!")


		# инициализировать веса
		for i in range(c):
			weights.append([random.randint(0, size), random.randint(0, size)])


		def classification_main(animator):
			global iteration, changed, edu_rate, edu_deteriorate

			def find_near(weights, xy):

				def get_d(weight, xy):
					d = (weight[0] - xy[0])**2
					d += (weight[1] - xy[1])**2
					return d

				weight = weights[0]
				d = get_d(weight, xy)
				i_n = 0

				for i, w in enumerate(weights):
					if get_d(w, xy) < d:  # победитель
						d = get_d(w, xy)
						weight = w
						i_n = i  # новый номер класса (нейрона)
				return weight, i_n

			if changed:
				check_distances(markers, id_of_cores)
				correcting()
				print('ISODATA: итерация', iteration)
				cores_data = [[], [], []]  # all x, all y, all colors
				points_data = [[], [], []]
				for elem in markers:
					if elem.klass_core:
						cores_data[0].append(elem.x)
						cores_data[1].append(elem.y)
						cores_data[2].append(elem.color)
					else:
						points_data[0].append(elem.x)
						points_data[1].append(elem.y)
						points_data[2].append(elem.color)
				ax1.clear()
				ax1.title.set_text("ISODATA")
				ax1.scatter(cores_data[0], cores_data[1], c=cores_data[2], alpha=0.8, marker='X')
				ax1.scatter(points_data[0], points_data[1], c=points_data[2], alpha=0.8, marker='.')
				ax1.axis([0, size + 1, 0, size + 1])
				if changed is False:
					print("ISODATA: ядра нашли свои места.")

			# Kohonen
			if edu_rate > 0.0:
				print('Сеть Кохонена: итерация', iteration)
				#for xy in XY:  # перебираем все точки
				xy = XY[random.randint(0, len(XY)-1)]  # асинхронный режим
				wm = find_near(weights, xy)[0]
				wm[0] += edu_rate * (xy[0] - wm[0])  # корректировка веса х
				wm[1] += edu_rate * (xy[1] - wm[1])  # корректировка весоа у
				edu_rate -= edu_deteriorate  # уменьшение коэффициента обучения

				kohonen_data = list()  # список для классов
				for i in range(len(weights)):
					kohonen_data.append(list())
				for xy in XY:
					i_n = find_near(weights, xy)[1]
					kohonen_data[i_n].append(xy)

				x_plt = []
				y_plt = []
				colors = []
				cores_x = []
				cores_y = []
				core_colors = []
				for i, cl in enumerate(kohonen_data):
					cores_x.append(weights[i][0])
					cores_y.append(weights[i][1])
					try:
						core_colors.append(Marker.COLORS[i])
					except IndexError:
						core_colors.append('#%02X%02X%02X' % (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
					if cl:
						for elem in cl:
							x_plt.append(elem[0])
							y_plt.append(elem[1])
							colors.append(core_colors[i])

				ax2.clear()
				ax2.title.set_text("Сеть Кохонена")
				ax2.scatter(x_plt, y_plt, c=colors, alpha=0.8, marker='.')
				ax2.scatter(cores_x, cores_y, c=core_colors, alpha=0.8, marker='X')
				ax2.axis([0, size + 1, 0, size + 1])
				if edu_rate <= 0.0:
					print("Сеть Кохонена: ядра нашли свои места.")

			if changed or edu_rate > 0.0:
				iteration += 1


		fig = plt.figure()
		ax1 = fig.add_subplot(121, title="ISODATA")
		ax2 = fig.add_subplot(122, title="Сеть Кохонена")
		ani = animation.FuncAnimation(fig, classification_main, interval=500)
		plt.show()
	elif selector == '2':
		mapsize = 100
		stops = 0
		mutationrate = -0.1
		maxmen = 0  # максимальное количество маршрутов (коммивояжёров)
		accuracy = 0

		#while mapsize < 100:
		#	print("Введите размер графика (больше 99): ")
		#	try:
		#		mapsize = int(input())
		#	except ValueError:
		#		print("Введите целочисленное значения: ")
		while stops < 4:
			print("Введите количество городов (минимум 4):")
			try:
				stops = int(input())
			except ValueError:
				print("Введите целочисленное значение!")

		print("ДЛЯ ГЕНЕТИЧЕСКОГО АЛГОРИТМА")
		while mutationrate < 0.0:
			print("Задайте коэффициент мутации:")
			try:
				mutationrate = float(input())
			except ValueError:
				print("Введите корректное значение!")
		while maxmen <= 1 or maxmen % 2 != 0:
			print("Введите количество маршрутов (кратно 2):")
			try:
				maxmen = int(input())
			except ValueError:
				print("Введите целочисленное значение!")
		while accuracy <= 0:
			print("Задайте точность (введите 1 для автоопределения):")
			try:
				accuracy = int(input())
			except ValueError:
				print("Введите целочисленное значение!")
		if accuracy == 1:
			accuracy = round(maxmen*25)
			print("Точность = %s" % accuracy)

		print("ДЛЯ МОНТЕ-КАРЛО")
		temperature = 0.0
		alpha = 0.0
		iterations = 0
		while stops < 4:
			print("Введите количество городов (минимум 4):")
			try:
				stops = int(input())
			except ValueError:
				print("Введите целочисленное значение!")
		while temperature < 2:
			print("Задайте температуру:")
			try:
				temperature = float(input())
			except ValueError:
				print("Введите корректное значение!")
		while alpha <= 0.0:
			print("Задайте скорость уменьшения температуры (больше 0):")
			try:
				alpha = float(input())
			except ValueError:
				print("Введите корректное значение!")
		while iterations < 2:
			print("Задайте количество итераций неизменности:")
			try:
				iterations = int(input())
			except ValueError:
				print("Введите целочисленное значение!")

		fig = plt.figure()
		ax1 = fig.add_subplot(121, title="Генетический алгоритм")
		ax2 = fig.add_subplot(122, title="Метрополис")

		men = Salesman(mapsize, stops, maxmen, mutationrate, temperature, alpha)

		# для ГА
		iteration = 1
		lens = men.calculate_lengths()
		print("LENS", lens)
		len_sum = abs(lens[len(lens)-1][1] - np.array(lens)[0][1])   # Разница длин лучшего и наихудшего коммивояжёров
		if len_sum > accuracy:
			while len_sum > accuracy:
				accuracy = round(accuracy*1.1)
			print("Точность была увеличена до ", accuracy)

		asked = False
		# для МК
		total_i = 0
		i = 0

		ani = animation.FuncAnimation(fig, men.calculate, interval=1)
		plt.show()
