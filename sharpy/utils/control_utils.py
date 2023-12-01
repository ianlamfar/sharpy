"""Controller Utilities
"""
import numpy as np

def second_order_fd(history, n_calls, dt):
    # history is ordered such that the last element is the most recent one (t), and thus
    # it goes [(t - 2), (t - 1), (t)]
    coefficients = np.zeros((3,))
    if n_calls <= 1:
        # no derivative, return 0
        pass

    elif n_calls == 2:
    # else:
        # first order derivative
        coefficients[1:3] = [-1, 1]

    else:
        # # second order derivative
        coefficients[:] = [1, -4, 3]
        coefficients *= 0.5

    derivative = np.dot(coefficients, history)/dt
    return derivative


def n_order_fd_1(history, n_calls, dt, order=2):
    coefficients = np.zeros((7,))
    n = min(order+1, n_calls)
    if n <= 1:
        pass

    elif n == 2:
        coefficients[-2:] = [-1, 1]
    
    elif n == 3:
        coefficients[-3:] = [1, -4, 3]
        coefficients /= 2
    
    elif n == 4:
        coefficients[-4:] = [-2, 9, -18, 11]
        coefficients /= 6
    
    elif n == 5:
        coefficients[-5:] = [3, -16, 36, -48, 25]
        coefficients /= 12
    
    elif n == 6:
        coefficients[-6:] = [-12, 75, -200, 300, -300, 137]
        coefficients /= 60
    
    elif n == 7:
        coefficients[-7:] = [10, -72, 225, -400, 450, -360, 147]
        coefficients /= 60
    
    else:
        raise Exception(f'Out of order, you should not arrive here, [n,order,n_calls] = [{n},{order},{n_calls}]')

    return np.dot(coefficients, history)/dt


def n_order_fd_2(history, n_calls, dt, order=2):
    coefficients = np.zeros((8,))
    n = min(order+2, n_calls)
    if n <= 2:
        pass

    elif n == 3:
        coefficients[-3:] = [1, -2, 1]
    
    elif n == 4:
        coefficients[-4:] = [-1, 4, -5, 2]
    
    elif n == 5:
        coefficients[-5:] = [11/12, -14/3, 19/2, -26/3, 35/12]
    
    elif n == 6:
        coefficients[-6:] = [-5/6, 61/12, -13, 107/6, -77/6, 15/4]
    
    elif n == 7:
        coefficients[-7:] = [137/180, -27/5, 33/2, -256/9, 117/4, -87/5, 203/45]
    
    elif n == 8:
        coefficients[-8:] = [-7/10, 1019/180, -201/10, 41, -949/18, 879/20, -223/10, 469/90]
    
    else:
        raise Exception(f'Out of order, you should not arrive here, [n,order,n_calls] = [{n},{order},{n_calls}]')

    # print(n, coefficients, history)

    return np.dot(coefficients, history)/(dt**2)


class PID(object):
    """
    Class implementing a classic PID controller

    Instance attributes:
    :param `gain_p`: Proportional gain.
    :param `gain_i`: Integral gain.
    :param `gain_d`: Derivative gain.
    :param `dt`: Simulation time step.

    The class should be used as:

        pid = PID(100, 10, 0.1, 0.1)
        pid.set_point(target_point)
        control = pid(current_point)
    """
    # for i in range(n_steps):
    #     state[i] = np.sum(feedback[:i])
    #     controller.set_point(set_point[i])
    #     feedback[i] = controller(state[i])
    def __init__(self, gain_p, gain_i, gain_d, dt):
        self._kp = gain_p
        self._ki = gain_i
        self._kd = gain_d

        self._point = 0.0

        self._dt = dt

        self._accumulated_integral = 0.0
        self._integral_limits = np.array([-1., 1.])*10000
        self._error_history = np.zeros((3,))

        self._derivator = second_order_fd
        self._derivative_limits = np.array([-1, 1])*10000

        self._n_calls = 0

        self._anti_windup_lim = None

    def set_point(self, point):
        self._point = point

    def set_anti_windup_lim(self, lim):
        self._anti_windup_lim = lim

    def reset_integrator(self):
        self._accumulated_integral = 0.0

    def __call__(self, state):
        self._n_calls += 1
        actuation = 0.0
        error = self._point - state
        # displace previous errors one position to the left
        self._error_history = np.roll(self._error_history, -1)
        self._error_history[-1] = error

        detailed = np.zeros((3,))
        # Proportional gain
        actuation += error*self._kp
        detailed[0] = error*self._kp

        # Derivative gain
        derivative = self._derivator(self._error_history, self._n_calls, self._dt)
        derivative = max(derivative, self._derivative_limits[0])
        derivative = min(derivative, self._derivative_limits[1])
        actuation += derivative*self._kd
        detailed[2] = derivative*self._kd

        # Integral gain
        aux_acc_int = self._accumulated_integral + error*self._dt
        if aux_acc_int < self._integral_limits[0]:
            aux_acc_int = self._integral_limits[0]
        elif aux_acc_int > self._integral_limits[1]:
            aux_acc_int = self._integral_limits[1]

        if self._anti_windup_lim is not None:
            # Apply anti wind up
            aux_actuation = actuation + aux_acc_int*self._ki
            if ((aux_actuation > self._anti_windup_lim[0]) and
                (aux_actuation < self._anti_windup_lim[1])):
                # Within limits
                self._accumulated_integral = aux_acc_int
                # If the system exceeds the limits, this 
                # will not be added to the self._accumulated_integral
        else:
            self._accumulated_integral = aux_acc_int

        actuation += self._accumulated_integral*self._ki
        detailed[1] = self._accumulated_integral*self._ki

        return actuation, detailed


class PDE(object):
    """
    Class implementing a PDE controller

    Instance attributes:
    :param `gain_p`: Proportional gain.
    :param `gain_i`: Integral gain.
    :param `gain_d`: Derivative gain.
    :param `dt`: Simulation time step.

    The class should be used as:

        pid = PID(100, 10, 0.1, 0.1)
        pid.set_point(target_point)
        control = pid(current_point)
    """
    # for i in range(n_steps):
    #     state[i] = np.sum(feedback[:i])
    #     controller.set_point(set_point[i])
    #     feedback[i] = controller(state[i])
    def __init__(self, gain_p, gain_i, gain_d, gain_d2, dt, order):
        self._point = 0.0

        self._dt = dt
        self._kp = gain_p
        self._ki = gain_i
        self._kd = gain_d
        self._kd2 = gain_d2
        
        self._order = order
        
        self._accumulated_integral = 0.0
        self._integral_limits = np.array([-1., 1.])*10000
        self._error_history = np.zeros((8,))

        self._derivator_1 = n_order_fd_1
        self._derivator_2 = n_order_fd_2
        self._derivative_limits = np.array([-1, 1])*10000

        self._n_calls = 0

        self._anti_windup_lim = None

    def set_point(self, point):
        self._point = point

    def set_anti_windup_lim(self, lim):
        self._anti_windup_lim = lim

    def reset_integrator(self):
        self._accumulated_integral = 0.0

    def __call__(self, state, power=1, rampups=[1, 1, 1, 1]):
        self._n_calls += 1
        actuation = 0.0
        error = self._point - state
        error = np.power(np.abs(error), power) * np.sign(error)
        # error = np.sqrt(np.abs(error)) * np.sign(error)
        
        # displace previous errors one position to the left
        self._error_history = np.roll(self._error_history, -1)
        self._error_history[-1] = error

        detailed = np.zeros((4,))
        
        # Proportional gain
        p_control = error*self._kp*min(1, (self._n_calls/max(1, abs(rampups[0]))))
        actuation += p_control
        detailed[0] = p_control

        # Integral gain
        aux_acc_int = self._accumulated_integral + error*self._dt
        aux_acc_int = min(max(aux_acc_int, self._integral_limits[0]), self._integral_limits[1])

        if self._anti_windup_lim is not None:
            # Apply anti wind up
            aux_actuation = actuation + aux_acc_int*self._ki
            if ((aux_actuation > self._anti_windup_lim[0]) and
                (aux_actuation < self._anti_windup_lim[1])):
                # Within limits
                self._accumulated_integral = aux_acc_int
                # If the system exceeds the limits, this 
                # will not be added to the self._accumulated_integral
        else:
            self._accumulated_integral = aux_acc_int

        i_control = self._accumulated_integral*self._ki*min(1, (self._n_calls/max(1, abs(rampups[1]))))
        actuation += i_control
        detailed[1] = i_control
        
        # First derivative gain
        derivative = self._derivator_1(self._error_history[1:], self._n_calls, self._dt, self._order)
        derivative = min(max(derivative, self._derivative_limits[0]), self._derivative_limits[1])
        d_control = derivative*self._kd*min(1, (self._n_calls/max(1, abs(rampups[2]))))
        actuation += d_control
        detailed[2] = d_control
        
        # Second derivative gain
        derivative_2 = self._derivator_2(self._error_history, self._n_calls, self._dt, self._order)
        derivative_2 = min(max(derivative_2, self._derivative_limits[0]), self._derivative_limits[1])
        d2_control = derivative_2*self._kd2*min(1, (self._n_calls/max(1, abs(rampups[3]))))
        actuation += d2_control
        detailed[3] = d2_control

        return actuation, detailed
