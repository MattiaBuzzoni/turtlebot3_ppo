from src.model.robots.turtlebot import constants, marks
from src.model.robots.robot import Robot


class TurtleBot3(Robot):

    #@classmethod
    #def GetMotorClass(cls):
    #    del cls
    #    return motor_constants.MOTOR_CONTROL_CLASS

    #@classmethod
    #def GetMotorConstants(cls):
    #    del cls
    #    return motor_constants

    #@classmethod
    #def GetCtrlConstants(cls):
    #   del cls
    #   return ctrl_constants

    @classmethod
    def get_constants(cls):
        del cls
        return constants

    @classmethod
    def get_marks(cls):
        del cls
        return marks
