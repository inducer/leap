program test_threerabmethod_squarewave
  use ThreeRAB, only: dagrt_state_type, &
    timestep_initialize => initialize, &
    timestep_run => run, &
    timestep_shutdown => shutdown

  implicit none

  type(dagrt_state_type), target :: state
  type(dagrt_state_type), pointer :: state_ptr

  real*8, dimension(3) :: initial_condition 
  real*8, dimension(1) :: true_sol_fast, true_sol_slow, true_sol_medium
  integer, dimension(2) :: ntrips

  integer run_count, k
  real*8 t_fin
  parameter (run_count=2, t_fin=11d0)

  real*8, dimension(run_count):: dt_values, error_slow, error_fast, error_medium

  real*8 min_order, est_order_fast, est_order_slow, est_order_medium

  integer stderr
  parameter(stderr=0)

  integer irun

  ! start code ----------------------------------------------------------------

  state_ptr => state

  initial_condition(1) = 1.0d0 ! fast
  initial_condition(2) = 1.0d0 ! slow
  initial_condition(3) = 1.0d0 ! medium

  ntrips(1) = NUM_TRIPS_ONE
  ntrips(2) = NUM_TRIPS_TWO

  do irun = 1,run_count
    dt_values(irun) = t_fin/ntrips(irun)

    call timestep_initialize( &
      dagrt_state=state_ptr, &
      state_slow=initial_condition(2:2), &
      state_fast=initial_condition(1:1), &
      state_medium=initial_condition(3:3), &
      dagrt_t=0d0, &
      dagrt_dt=dt_values(irun))

    k = 0

    do
      if (k == 1) then
        state%dagrt_dt = dt_values(irun)/4
        k = 0
      else
        state%dagrt_dt = dt_values(irun)
        k = 1
      endif
      call timestep_run(dagrt_state=state_ptr)
      if (state%ret_time_fast.ge.t_fin) then
        exit
      endif
    end do 

    true_sol_fast = exp(-20.0*state%ret_time_fast)
    true_sol_slow = exp(-2.0*state%ret_time_slow)
    true_sol_medium = exp(-10.0*state%ret_time_medium)

    error_slow(irun) = sqrt(sum((true_sol_slow-state%ret_state_slow)**2))
    error_fast(irun) = sqrt(sum((true_sol_fast-state%ret_state_fast)**2))
    error_medium(irun) = sqrt(sum((true_sol_medium-state%ret_state_medium)**2))

    call timestep_shutdown(dagrt_state=state_ptr)
    write(*,*) 'done', dt_values(irun), error_slow(irun), error_fast(irun), error_medium(irun)
  enddo

  min_order = MIN_ORDER
  est_order_slow = log(error_slow(2)/error_slow(1))/log(dt_values(2)/dt_values(1))
  est_order_fast = log(error_fast(2)/error_fast(1))/log(dt_values(2)/dt_values(1))
  est_order_medium = log(error_medium(2)/error_medium(1))/log(dt_values(2)/dt_values(1))

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

  write(*,*) 'estimated order medium:', est_order_medium
  if (est_order_medium < min_order) then
    write(stderr,*) 'ERROR: achieved order too low:', est_order_medium, ' < ', &
        min_order
  endif

end program
