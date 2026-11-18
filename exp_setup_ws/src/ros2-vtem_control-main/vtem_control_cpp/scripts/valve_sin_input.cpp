#include "VtemControl.hpp"
#include <chrono>
#include <ctime>
#include <iostream>
#include <thread>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <random>
#include <vector>


#define AMPL_UPPER_BOUND 290
#define AMPL_LOWER_BOUND 0
#define FREQ_UPPER_BOUND 25
#define FREQ_LOWER_BOUND 5


/*
   Test program for actuating a single valve.
*/
int step_func() {
    static int step = AMPL_LOWER_BOUND;
    step += 1;
    if (step / AMPL_UPPER_BOUND >= 1)
        step -= 1 ;
    return step;
}

int inv_step_func() {
    static int step = AMPL_UPPER_BOUND;
    step -= 1;
    if (step - AMPL_LOWER_BOUND <= 0)
        step += 1 ;
    return step;
}

int trapezoid_func() {
    static int step = AMPL_LOWER_BOUND;
    static bool flag = 0;
    static int timer = 0;
    static bool big_flag = 0;
    if (big_flag)
    {
        step = AMPL_LOWER_BOUND;
    }
    else {
        if (flag == 0) // RAMP UP
        {
            step++;

            if (step >= AMPL_UPPER_BOUND/2)
            {
                step = AMPL_UPPER_BOUND/2;
                flag = 1;
                timer = 0;
            }
        }
        else // FLAG == 1
        {
            if (timer < 50) 
            {
                timer++;
            }
            else // RAMP DOWN
            {
                step--;

                if (step <= 0)
                {
                    step = 0;
                    flag = 0;
                    timer = 0;
                    big_flag = 1;
                }
            }
        }
    }
    

    return step;
}


int rec_trapezoid_func(void)
{
    static int step = AMPL_LOWER_BOUND;
    static bool flag = 0;   // 0 = ramp up, 1 = hold/ramp down
    static int timer = 0;

    if (flag == 0) // RAMP UP
    {
        step++;

        if (step >= AMPL_UPPER_BOUND / 2)
        {
            step = AMPL_UPPER_BOUND / 2;
            flag = 1;
            timer = 0;
        }
    }
    else // HOLD THEN RAMP DOWN
    {
        if (timer < 50)
        {
            timer++;
        }
        else
        {
            step--;

            if (step <= AMPL_LOWER_BOUND)
            {
                step = AMPL_LOWER_BOUND;
                flag = 0;
                timer = 0;
            }
        }
    }

    return step;
}



int triang_func(){
    static int step = 0;
    step += 1;
    int output;
    if (( step + 1 ) > 500)
        output = AMPL_UPPER_BOUND - AMPL_UPPER_BOUND * (step - 500) / 500.0;
    else
        output = AMPL_UPPER_BOUND*step/500.0;
    return output;
}

int sine_input(){
    static int step = 0;
    step += 1;
    double omega = 0.05 +  0.05*step/200;// f = 0.05
    return int(175+ 175*sin(2*M_1_PI*omega*step)); // T = 1/0.0.5= 20s.  Tinit = 20s ==> T_end = 10s : 
}

int main() {
    // Create VtemControl controller.
    vtem_control::VtemControl vtemControl("192.168.4.3", "502");

    unsigned int valveIdx = 2;          // test valves 0 to 15
    unsigned int slotIdx = vtemControl.get_slot_idx_from_valve_idx(valveIdx); 
    
    unsigned int commandedPressure = 2; // tested 0 to 450 mbar
    // float commandedPressureFloat = 1.5; // for sinusoidal input

    // Connect.
    if (!vtemControl.connect()) {
        std::cout << "Failed to connect to VTEM." << std::endl;
        return -1;
    }

    // acknowledge errors for all slots
    vtemControl.acknowledge_errors();

    // Set motion app for all valves to 03 (proportional pressure regulation)
    if (!vtemControl.activate_pressure_regulation(slotIdx)) {
        std::cout << "Failed to activate pressure regulation." << std::endl;
        return -1;
    }
    std::ofstream logfile("pressure_log.csv");
    if (!logfile.is_open()) {
        std::cerr << "Failed to open log file!" << std::endl;
        return -1;
    }

    // CSV header
    logfile << "time_s,commanded_mbar,actual_mbar\n";
    logfile << std::fixed << std::setprecision(3);

    // vtemControl.set_single_pressure(valveIdx, commandedPressure);
    int actualPressure;
    int cycles = 10000;
    std::vector<double> x(cycles), actualPressures(cycles),
            commandedPressures(cycles); // for logging pressure profile

    time_t tstart, tend;
    tstart = time(0);
    uint64_t t_sleep = 200; // time in millisecods

    // make random pertubation signal. Obtained from : https://stackoverflow.com/questions/13006929/pseudo-random-binary-sequences-prbs-in-python
    std::vector<int> amplitude;
    std::vector<unsigned int> freq;

    amplitude.reserve(cycles);
    freq.reserve(cycles);

    srand(static_cast<unsigned>(time(0)));

    for (int i = 0; i < cycles; ++i) {
        float tempA = static_cast<float>(rand()) / RAND_MAX;
        float tempF = static_cast<float>(rand()) / RAND_MAX;

        amplitude.push_back(static_cast<int>(
            std::lround(tempA * (AMPL_UPPER_BOUND - AMPL_LOWER_BOUND) + AMPL_LOWER_BOUND)
        ));

        freq.push_back(static_cast<unsigned int>(
            std::lround(tempF * (FREQ_UPPER_BOUND - FREQ_LOWER_BOUND) + FREQ_LOWER_BOUND)
        ));
    }

    freq[0] = 0;
    for (size_t i = 1; i < freq.size(); ++i) {
        freq[i] = freq[i - 1] + freq[i];
    }

    std::vector<int> random_signal(cycles, 0);

    size_t i = 0;
    while (i < freq.size() && freq[i] < random_signal.size()) {
        size_t k = freq[i];
        std::fill(random_signal.begin() + k, random_signal.end(), amplitude[i]);
        ++i;
    }
    
    for (int i = 0; i < cycles; i++) {
        // Wait 200 ms.
        std::this_thread::sleep_for(std::chrono::milliseconds(t_sleep));

        commandedPressure = rec_trapezoid_func();
        // commandedPressure = step_func();
        // commandedPressure = inv_step_func();
        // commandedPressure = trapezoid_func();
        // commandedPressure = random_signal.at(i);
        // commandedPressure = sine_input();
        
        commandedPressures.at(i) = commandedPressure;
        // Read pressure of valve 0.
        actualPressure = vtemControl.get_single_pressure(valveIdx);
        x.at(i) = i;
        actualPressures.at(i) = actualPressure;
        // Log to file
        logfile << commandedPressure << ","
                << actualPressure << "\n";

        vtemControl.set_single_pressure(valveIdx, commandedPressure);

        if (i % 50 == 0) {
            std::cout << " Valve " << valveIdx << " actual pressure: " << actualPressure
                        << " mBar\tcommanded pressure: " << commandedPressure << " mBar" << std::endl;
        }
    }

    tend = time(0);
    std::cout << "Average computational duration of cycle: " << 1000*difftime(tend, tstart) / ((double) (cycles)) << " ms" << std::endl;

    if (!vtemControl.deactivate_pressure_regulation(slotIdx)) {
        std::cout << "Failed to deactivate pressure regulation." << std::endl;
        return -1;
    }

    // Disconnect.
    vtemControl.disconnect();
    logfile.close();

    return 0;
}