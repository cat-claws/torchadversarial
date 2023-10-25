import torch
from torch import Tensor

from torch.optim.optimizer import (Optimizer, _use_grad_for_differentiable, _default_to_fused_or_foreach,
                        _differentiable_doc, _foreach_doc, _maximize_doc)
from typing import List, Optional

__all__ = ["Fgsm", "fgsm"]


class Fgsm(Optimizer):
	def __init__(
		self,
		params,
		epsilon=8/255,
		a_min=0,
		a_max=1,
		foreach: Optional[bool] = None,
		*,
		maximize: bool = True,
		differentiable: bool = False,
	):
		# if lr is not required and lr <= 0.0:
		#     raise ValueError(f"Invalid learning rate: {lr}")
		# if epsilon is not required and epsilon <= 0.0:
		#     raise ValueError(f"Invalid epsilon value: {epsilon}")
	
		defaults = dict(
			epsilon=epsilon,
			a_min=a_min,
			a_max=a_max,
			maximize = maximize,
			foreach=foreach,
			differentiable=differentiable,
		)
		super().__init__(params, defaults)
	
	def __setstate__(self, state):
		super().__setstate__(state)
		for group in self.param_groups:
			group.setdefault("foreach", None)
			# group.setdefault("maximize", True)
			group.setdefault("differentiable", False)
	
	def _init_group(self, group, params_with_grad, grads, acc_deltas):
		for p in group["params"]:
			if p.grad is None:
				continue
			params_with_grad.append(p)
			grads.append(p.grad)
	
			state = self.state[p]
	
			# Lazy state initialization
			if len(state) == 0:
				state["step"] = 0
				state["acc_delta"] = torch.zeros_like(
					p, memory_format=torch.preserve_format
				)
	
			acc_deltas.append(state["acc_delta"])
	
			state["step"] += 1
	
	@_use_grad_for_differentiable
	def step(self, closure=None):
		loss = None
		if closure is not None:
			with torch.enable_grad():
				loss = closure()
	
		for group in self.param_groups:
			params_with_grad = []
			grads = []
			acc_deltas = []
			epsilon, a_min, a_max, foreach, maximize, differentiable = (
				group["epsilon"],
				group["a_min"],
				group["a_max"],
				group["foreach"],
				group["maximize"],
				group["differentiable"],
			)
	
			self._init_group(group, params_with_grad, grads, acc_deltas)
	
			fgsm(
				params_with_grad,
				grads,
				acc_deltas,
				epsilon = epsilon,
				a_min = a_min,
				a_max = a_max,
				foreach=foreach,
				maximize=maximize,
				differentiable=differentiable,
			)
	
		return loss


def fgsm(
	params: List[Tensor],
	grads: List[Tensor],
	acc_deltas: List[Tensor],
	foreach: Optional[bool] = None,
	differentiable: bool = False,
	*,
	epsilon: float,
	a_min: Optional[float] = 0,
	a_max: Optional[float] = 1,
	maximize: bool = True,
):
	r"""Functional API that performs FGSM algorithm computation.
	
	See :class:`~torch.optim.-` for details.
	"""
	
	# We still respect when the user inputs False for foreach.
	if foreach is None:
		_, foreach = _default_to_fused_or_foreach(params, differentiable, use_fused=False)
	
	if foreach and torch.jit.is_scripting():
		raise RuntimeError("torch.jit.script not supported with foreach optimizers")
	
	if foreach and not torch.jit.is_scripting():
		func = '_multi_tensor_fgsm not exist'
	else:
		func = _single_tensor_fgsm
	
	func(
		params,
		grads,
		acc_deltas,
		a_min = a_min,
		a_max = a_max,
		epsilon = epsilon,
		maximize=maximize,
		differentiable=differentiable,
	)

def _single_tensor_fgsm(
	params: List[Tensor],
	grads: List[Tensor],
	acc_deltas: List[Tensor],
	*,
	epsilon: float,
	a_min: Optional[float] = 0,
	a_max: Optional[float] = 1,
	foreach: Optional[bool] = False,
	maximize: bool = True,
	differentiable: bool = True,
):
	for param, grad, acc_delta in zip(params, grads, acc_deltas):
		
		grad = grad.sign() if not maximize else -grad.sign()
		delta = grad * epsilon
		delta = torch.clamp(delta, - epsilon - acc_delta, epsilon - acc_delta)
		delta = torch.clamp(delta, a_min - param, a_max - param)
	
		acc_delta.add_(delta)
		param.add_(delta)
