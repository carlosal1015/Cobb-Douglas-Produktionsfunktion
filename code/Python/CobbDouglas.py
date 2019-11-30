#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
from statistics import mean, variance
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def plotCD(fig, data, reg1, reg2, log):
	"""
	Método responsable de hacer el trazado de las superficies de regresión.
	Se recomienda establecer el divisor del intervalo con la correspondencia con los datos iniciales.
	"""
	interval = (max(data["K"]) - min(data["K"])) // 20
	interval2 = (max(data["L"]) - min(data["L"])) // 20
	x = np.arange(min(data["K"]), max(data["K"]), interval)
	y = np.arange(min(data["L"]), max(data["L"]), interval2)
	x, y = np.meshgrid(x, y)
	fig.suptitle('Función de producción Cobb-Douglas')
	z1 = (math.exp(reg1[0]) if not log else reg1[0]) * x ** reg1[1] * y ** (1 - reg1[1])
	z2 = (math.exp(reg2[0]) if not log else reg2[0]) * x ** reg2[1] * y ** reg2[2]
	z = [z1,z2]
	for i in range (2):
		ax = fig.add_subplot(1, 2, i+1, projection='3d')
		ax.plot_wireframe(x, y, z[i], antialiased=False, rstride=2, cstride=2, color="green" if i==0 else "blue", linewidth=1)
		ax.set_title("Rendimientos de escala constante" if i==0 else "Rendimientos de escala variable", fontweight="bold")
		ax.set_xlabel('K', fontweight="bold")
		ax.set_ylabel('L', fontweight="bold")
		ax.set_zlabel('Y', fontweight="bold")
		handles, labels = ax.get_legend_handles_labels()
		ax.legend(handles, labels)
		red_patch = mpatches.Patch(color='red', label='Puntos de datos iniciales')
		plot_patch = mpatches.Patch(color="orange" if i==0 else "blue", label="Superficie de regressión")
		plt.legend(handles=[red_patch, plot_patch])
		ax.scatter(data["K"], data["L"], data["Y"], c="red", linewidth=0, antialiased=False)
	plt.show()

def getData(file, log, d=';'):
	"""
	Obtenga el archivo de datos csv basado en el nombre
	"""
	data = {"Y": [],
			"K": [],
			"L": [],
			"P": []}

	with open(file, 'r', newline='') as csvfile:
		freader = csv.reader(csvfile, delimiter=d)
		next(freader)
		for row in freader:
			if (not log):
				row = [np.log(np.float(n.replace(",", "."))) for n in row]
			else:
				row = [float(n.replace(",", ".")) for n in row]
			data["Y"].append(row[0])
			data["K"].append(row[1])
			data["L"].append(row[2])
			if (len(row) > 3): data["P"].append(row[3])
	return data

class RegressionModel:
	y = 0
	x1 = []
	x2 = None
	x3 = None
	residuals = []
	file = ""
	log = False
	model = []
	cond = 0


	def __init__(self, y, x1, x2=None, x3=None):
		self.y = y
		self.x1 = x1
		self.x2 = x2
		self.x3 = x3

	def cov(self, a, b):
		"""
		Método para calcular para calcular la covarianza
		"""
		cov = 0.0
		for i in range(len(a)):
			cov += (a[i] - mean(a)) * (b[i] - mean(b))
		return cov / (len(a) - 1)

	def se(self, y, x1, residuals, x2=None, x3=None):
		"""
		Errores estándar
		"""
		se = []
		SSr = sum([(res) ** 2 for res in residuals])
		MSE = SSr / (len(y) - 3)
		if (x2 is None):
			s = (sum([res ** 2 for res in residuals]) / (len(y) - 2)) ** 0.5
			SSX = sum([(x - mean(x1)) ** 2 for x in x1])
			xsq = [x ** 2 for x in x1]
			se.append(s * (sum(xsq) / (len(y) * SSX)) ** 0.5)
			se.append(s / (SSX) ** 0.5)
			return se
		elif (x3 is None):
			mat = np.column_stack((np.array(np.ones(len(y))), np.array(x1), np.array(x2)))
		else:
			mat = np.column_stack((np.array(np.ones(len(y))), np.array(x1), np.array(x2), np.array(x3)))
		mat = np.linalg.pinv(np.matmul(mat.transpose(), mat))
		se = [(d * MSE) ** 0.5 for d in mat.diagonal()]
		return se

	def getRes(self, y, x1, b0, b1, x2=None, b2=None, x3=None, b3=None):
		"""
		Obtener los residuos de la regresión calculada.
		"""
		res = []
		yp = []
		if (x2 is None):
			for i in range(len(y)):
				yp.append(b0 + b1 * x1[i])
				res.append(y[i] - yp[i])
		elif (x3 is None):
			for i in range(len(y)):
				yp.append(b0 + b1 * x1[i] + b2 * x2[i])
				res.append(y[i] - yp[i])
		else:
			for i in range(len(y)):
				yp.append(b0 + b1 * x1[i] + b2 * x2[i] + b3 * x3[i])
				res.append(y[i] - yp[i])
		return res, yp

	def r2(self, y, residuals, ym):
		"""
		Coefficient of determination
		"""
		SSr = sum([res ** 2 for res in residuals])
		SSt = sum([(yi - ym) ** 2 for yi in y])
		return 1 - (SSr / SSt) if SSt !=0 else 1

	def r2_adj(self, y, R2, fac):
		"""
		Coeficiente de determinación (ajustado)
		"""
		return 1 - (1 - R2) * ((len(y) - 1) / (len(y) - fac - 1))

	def f(self, y, yp, R2, fac):
		"""
		Prueba F
		"""
		SSE = 0.0
		SSM = 0.0
		for i in range(len(y)):
			SSE += (y[i] - yp[i]) ** 2
			SSM += (yp[i] - mean(y)) ** 2
		return (SSM / (fac)) / (SSE / (len(y) - fac - 1)) if SSE != 0 else math.inf

	def t(self, coeff, se):
		"""
		t-statistic
		"""
		t_stat = []
		for i in range(len(coeff)):
			if se[i] ==0:
				continue
			t_stat.append(coeff[i] / se[i])
		return t_stat

	def dw(self, residuals):
		"""
		Durbin-Watson criteria
		"""
		sumr = 0.0
		rsq = sum([res ** 2 for res in residuals])
		for i in range(1, len(residuals)):
			sumr += (residuals[i] - residuals[i - 1]) ** 2
		return sumr / rsq if rsq !=0 else 0

	def jb(self, y, residuals):
		"""
		Jarque-Bera test
		"""
		m3 = sum([res ** 3 for res in residuals]) / len(y)
		sig3 = (sum([res ** 2 for res in residuals]) / len(y)) ** 1.5
		m4 = sum([res ** 4 for res in residuals]) / len(y)
		sig4 = (sum([res ** 2 for res in residuals]) / len(y)) ** 2
		S = m3 / sig3 if sig3 !=0 else 0
		C = m4 / sig4 if sig3 !=0 else 0
		jb_stat = len(y) * ((S ** 2) / 6 + ((C - 3) ** 2) / 24)
		return jb_stat

	def regr(self, y, x1, x2=None, x3=None):
		"""
		Método para calcular los coeficientes de regresión.
		"""
		if x2 is None:
			b1 = self.cov(x1, y) / variance(x1)
			b0 = mean(y) - b1 * mean(x1)
			coeff = [b0, b1]
			return coeff
		elif x3 is None:
			X = np.column_stack((np.array(np.ones(len(y))), np.array(x1), np.array(x2)))
		else:
			X = np.column_stack((np.array(np.ones(len(y))), np.array(x1), np.array(x2), np.array(x3)))
		Y = np.column_stack(np.array(y))
		A = np.linalg.inv(np.matmul(X.transpose(), X))
		B = np.matmul(X.transpose(), Y.transpose())
		coeff = np.matmul(A, B)
		self.cond = np.linalg.cond(np.matmul(X.transpose(), X))
		coeff = np.squeeze(np.asarray(coeff))
		return coeff

	def CD(self):
		"""
		Método principal para el cálculo de regresión y estadísticas.
		"""
		y = self.y
		x1 =self.x1
		x2 = self.x2
		x3 = self.x3
		model = self.regr(y, x1, x2, x3)
		if len(model) == 3:
			res, yp = self.getRes(y, x1, model[0], model[1], x2, model[2])
		elif len(model) == 2:
			res, yp = self.getRes(y, x1, model[0], model[1])
		else:
			res, yp = self.getRes(y, x1, model[0], model[1], x2, model[2], x3, model[3])

		R2 = self.r2(y, res, mean(y))
		R2_adj = self.r2_adj(y, R2, len(model) - 1)
		dw_test = self.dw(res)
		F = self.f(y, yp, R2, len(model) - 1)
		SE = self.se(y, x1, res, x2, x3)
		t_stat = self.t(model, SE)
		jb_test = self.jb(y, res)
		self.model = model
		res = {"Coeficientes de regresión": model,
			   "Errores estándar": SE,
			   "t-statistic": t_stat,
			   "Coeficiente de determinación": R2,
			   "Coeficiente de determinación (ajustado)": R2_adj,
			   "Prueba F": F,
			   "Estadístico de Durbin-Watson": dw_test,
			   "Test de Jarque-Bera": jb_test,
			   "Número de condición de X^tX": self.cond}

		names_stat = ["Coeficientes de regresión", "Errores estándar", "t-statistic", "Coeficiente de determinación", "Coeficiente de determinación (ajustado)"
			, "Prueba F", "Estadístico de Durbin-Watson", "Test de Jarque-Bera","Número de condición de X^tX"]
		print("{0}\n{1:^103}\n{2}".format("=" * 103, "Resumen de la regresión", "=" * 103))
		for i in range(len(names_stat)):
			print("{0:40} {1:}".format(names_stat[i], res[names_stat[i]]))
		print("\n")
		return res

def model():
	"""
	CLI interface
	"""
	while(True):
		try:
			file = input("Especifique el nombre del archivo de datos: ")
			ans = input("¿Aplicar logaritmo natural? (0-SÍ, 1-NO): ")
			while (ans not in ("1", "0")):
				print("Input 0 for YES and 1 for NO!\n")
				ans = input("¿Aplicar logaritmo natural? (0-SÍ, 1-NO): ")
			log = ans == "1"
			data = getData(file, log)
			fig = plt.figure()
			if (len(data["P"]) !=0):
				reg3 = RegressionModel([a - b for a, b in zip(data["Y"], data["P"])],
									   [a - b for a, b in zip(data["K"], data["P"])],
									   [a - b for a, b in zip(data["L"], data["P"])])
				reg4 = RegressionModel(data["Y"], data["K"], data["L"], data["P"])
				reg3.CD()
				reg4.CD()
			else:
				reg1 = RegressionModel([a - b for a, b in zip(data["Y"], data["L"])],
									   [a - b for a, b in zip(data["K"], data["L"])])
				reg2 = RegressionModel(data["Y"], data["K"], data["L"])
				reg1.CD()
				reg2.CD()
				plotCD(fig, getData(file, True), reg1.model, reg2.model, log)
		except Exception as err:
			print(err,"\n")
			continue

model()