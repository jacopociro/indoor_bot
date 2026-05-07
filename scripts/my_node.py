#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist




def publish_velocity():
    # Inizializza il nodo ROS
    rospy.init_node('velocity_publisher', anonymous=True)
    
    # Crea un publisher per il topic /cmd_vel
    velocity_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    
    # Imposta la frequenza di pubblicazione
    rate = rospy.Rate(10)  # 10 Hz
    
    # Crea un messaggio Twist
    vel_msg = Twist()
    vel_msg.linear.x = 0.5  # Velocita lineare in avanti
    vel_msg.angular.z = 0.2  # Velocita angolare
    
    rospy.loginfo("Inizio pubblicazione velocità su /cmd_vel")
    
    while not rospy.is_shutdown():
        # Pubblica il messaggio
        velocity_publisher.publish(vel_msg)
        rate.sleep()

if __name__ == '__main__':
    try:
        publish_velocity()
    except rospy.ROSInterruptException:
        pass