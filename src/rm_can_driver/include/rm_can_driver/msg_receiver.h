#ifndef _SERIAL_RECEIVER_H
#define _SERIAL_RECEIVER_H

#include <rm_utils/datatypes.h>
#include <stdint.h>
#include <rclcpp/rclcpp.hpp>
#include <functional>
template <typename T>
class msg_receiver {
   public:
    typedef std::function<void(T*)> dataready_callback;
    uint8_t data_updated;
    T* recv_data;

    msg_receiver(const dataready_callback& _callback, const rclcpp::Logger& _logger): logger(_logger) {
        callback = _callback;
        data_len = sizeof(T);
        // s/e 2 位 len 1 位 crc 2位
        buf_len = data_len + 5;
        rx_buf.resize(buf_len);
        recv_len = recv_status = data_updated = 0;
        recv_once = 0;
        data_rx.data.resize(data_len);
        recv_data = (T*)data_rx.data.data();
    }
    void receive(uint8_t* data, int len) {
        if (len <= 0){
            RCLCPP_WARN(logger,"[CAN_RECV:%s] Invaild data len: %d!",typeid(T).name(),len);
            return;
        }
        if(data[0] == 's'){
            recv_status = 1;
        }
        if(recv_status){
            if(recv_len + len > buf_len){
                RCLCPP_WARN(logger,"[CAN_RECV:%s] package length exceeded: %d!",typeid(T).name(),recv_len + len);
                recv_status = 0;
                recv_len = 0;
                return;
            }
            memcpy(rx_buf.data()+ recv_len, data, len);
            recv_len += len;
            if(recv_len == buf_len && rx_buf[recv_len - 1] == 'e'){
                if(buffer_check_valid(rx_buf.data() + 1, buf_len - 2,CRC16::crc16_ccitt)){
                    data_rx.read_buffer(rx_buf.data() + 1);
                    recv_status = 0;
                    recv_len = 0;
                    data_updated = 1;
                    if(callback){
                        callback(recv_data);
                    }
                }
            }
        }
    }

   private:
    uint8_t recv_status;
    uint8_t recv_once;

    uint8_t data_len;
    uint8_t buf_len;
    uint8_t recv_len;
    dataready_callback callback;
    general_data data_rx;
    std::vector<uint8_t> rx_buf;
    rclcpp::Logger logger;
};

#endif