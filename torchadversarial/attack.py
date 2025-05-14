from collections.abc import Iterator

class Attack(Iterator):
	def __init__(
		self,
		optimizer,
		params,
		**kwargs,
	):
		self._index = 0
		self.params = [param.detach().clone().requires_grad_() for param in params]
		self.steps = kwargs.pop('steps', 1)
		self.optimizer = optimizer(self.params, **kwargs)
	
	def __next__(self):
		if self._index > 0:
			self.optimizer.step()
		self.optimizer.zero_grad()
		if self._index < self.steps:
			self._index += 1
			return self.params
		else:
			for param in self.params:
				param.requires_grad = False
			raise StopIteration
