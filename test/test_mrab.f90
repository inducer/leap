program test_mrabmethod
  use MRAB, only: dagrt_state_type, &
    timestep_initialize => initialize, &
    timestep_run => run, &
    timestep_shutdown => shutdown

  implicit none

  type(dagrt_state_type), target :: state
  type(dagrt_state_type), pointer :: state_ptr

  real*8, dimension(2) :: initial_condition 
  real*8, dimension(1) :: true_sol_fast, true_sol_slow
  integer, dimension(2) :: ntrips

  integer run_count
  real*8 t_fin
  parameter (run_count=2, t_fin=50d-0)

  real*8, dimension(run_count):: dt_values, error_slow, error_fast

  real*8 min_order, est_order_fast, est_order_slow
  real*8 err_slow, err_fast
  real*8 max_err_slow, max_err_fast
  real*8 max_val_slow, max_val_fast

  integer stderr
  parameter(stderr=0)

  integer irun

  ! start code ----------------------------------------------------------------

  state_ptr => state

  initial_condition(1) = (exp(0d0))*cos(0d0) ! fast
  initial_condition(2) = (exp(0d0))*sin(0d0) ! slow

  ntrips(1) = NUM_TRIPS_ONE
  ntrips(2) = NUM_TRIPS_TWO

  do irun = 1,run_count
    dt_values(irun) = t_fin/ntrips(irun)

    call timestep_initialize( &
      dagrt_state=state_ptr, &
      state_slow=initial_condition(2:2), &
      state_fast=initial_condition(1:1), &
      dagrt_t=0d0, &
      dagrt_dt=dt_values(irun))

    max_err_fast = 0
    max_val_fast = 0
    max_err_slow = 0
    max_val_slow = 0

    do
      call timestep_run(dagrt_state=state_ptr)

      if (.not.isnan(state%ret_time_fast)) then
        true_sol_fast = (exp(state%ret_time_fast))*cos(state%ret_time_fast)
        true_sol_slow = (exp(state%ret_time_slow))*sin(state%ret_time_slow)

        err_slow = abs(true_sol_slow(1) - state%ret_state_slow(1))
        err_fast = abs(true_sol_fast(1) - state%ret_state_fast(1))

        max_err_slow = max(max_err_slow, err_slow)
        max_err_fast = max(max_err_fast, err_fast)

        max_val_slow = max(max_val_slow, abs(true_sol_slow(1)))
        max_val_fast = max(max_val_fast, abs(true_sol_fast(1)))
        !write(27,*) state%ret_time_slow, err_slow, true_sol_slow
        !write(28,*) state%ret_time_fast, err_fast, true_sol_fast

      endif

      if (state%ret_time_fast.ge.t_fin - 1d-10) then
        exit
      endif
    end do 
    !write(27,*) ''
    !write(28,*) ''

    write(*,*) 'end time slow:', state%ret_time_slow
    write(*,*) 'end time fast:', state%ret_time_fast

    error_slow(irun) = max_err_slow/max_val_slow
    error_fast(irun) = max_err_fast/max_val_fast

    call timestep_shutdown(dagrt_state=state_ptr)
    write(*,*) 'done', dt_values(irun), '  ', &
      'err_slow: ', error_slow(irun), '  ', &
      'err_fast: ', error_fast(irun)
  enddo

  min_order = MIN_ORDER
  est_order_slow = log(error_slow(2)/error_slow(1))/log(dt_values(2)/dt_values(1))
  est_order_fast = log(error_fast(2)/error_fast(1))/log(dt_values(2)/dt_values(1))

  write(*,*) 'estimated order slow:', est_order_slow
  if (est_order_slow < min_order) then
    write(stderr,*) 'ERROR: achieved order too low:', est_order_slow, ' < ', &
        min_order
  endif

  write(*,*) 'estimated order fast:', est_order_fast
  if (est_order_fast < min_order) then
    write(stderr,*) 'ERROR: achieved order too low:', est_order_fast, ' < ', &
        min_order
  endif

end program
