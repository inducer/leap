module timing
  contains
  function get_time()
    implicit none
    integer(kind=8) :: cnt, cnt_rate, cnt_max
    real*8 get_time

    call system_clock(cnt, cnt_rate, cnt_max)
    get_time = cnt
    get_time = get_time/cnt_rate
  end function
end module
